import os
import datetime
import bisect
import numpy as np
#import cv2
import pandas as pd
import logging
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, JointState
#from cv_bridge import CvBridge
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image as PILImage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#_bridge = CvBridge()

ACTION_RATE = 20


def frame_is_valid(frame):
    return (
        frame["image"] is not None
        and frame["wrist_image"] is not None
        and frame["state"] is not None
        and len(frame["actions"]) >= 1
    )


def parse_image(msg, target_size=(224, 224)):
    """
    Deserialize sensor_msgs/Image and return a HxWx3 uint8 RGB numpy array resized to target_size.
    Supports common encodings: rgb8, bgr8, rgba8, bgra8, mono8.
    """
    msg = deserialize_message(msg, Image)

    # create 1D view of the raw bytes and reshape to (H, W, C) using step
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    # step is bytes per row; channels = step / width
    channels = int(msg.step // msg.width) if msg.width > 0 else 1
    try:
        arr = arr.reshape((msg.height, msg.width, channels))
    except Exception:
        # fallback: try (H, W) grayscale
        arr = arr.reshape((msg.height, msg.width))

    # normalize channel layout to RGB 3-channel uint8
    if msg.encoding == "bgr8":
        arr = arr[..., ::-1]
    elif msg.encoding == "bgra8":
        arr = arr[..., [2, 1, 0, 3]][..., :3]
    elif msg.encoding == "rgba8":
        arr = arr[..., :3]
    elif msg.encoding == "rgb8":
        arr = arr[..., :3]
    elif msg.encoding == "mono8":
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = np.stack([arr, arr, arr], axis=-1)
    else:
        # Unknown encoding: try to convert to RGB via PIL after treating buffer as bytes
        try:
            pil_try = PILImage.frombuffer("RGB", (msg.width, msg.height), msg.data, "raw", msg.encoding, msg.step, 1)
            pil = pil_try.convert("RGB").resize(target_size, resample=PILImage.BILINEAR)
            return np.asarray(pil)
        except Exception:
            raise RuntimeError(f"Unsupported/unknown image encoding: {msg.encoding}")

    # Use PIL for resizing (bilinear)
    pil = PILImage.fromarray(arr)
    pil = pil.convert("RGB").resize(target_size, resample=PILImage.BILINEAR)
    return np.asarray(pil)


def parse_droid_joint_state(msg, joint_reordering=(0, 1, 2, 3, 4, 5, 6, 7)):
    msg = deserialize_message(msg, JointState)
    return [
        msg.position[joint_reordering[0]],
        msg.position[joint_reordering[1]],
        msg.position[joint_reordering[2]],
        msg.position[joint_reordering[3]],
        msg.position[joint_reordering[4]],
        msg.position[joint_reordering[5]],
        msg.position[joint_reordering[6]],
        msg.position[joint_reordering[7]],
    ]


def iterate_bag(bag: SequentialReader):
    while bag.has_next():
        topic, msg, t = bag.read_next()
        yield topic, msg, t

def reorder_bag_timestamps(bag: SequentialReader, *, time_source: str = "record"):
    """Yield messages sorted by a chosen time source.

    Args:
        bag: Open rosbag2 SequentialReader
        time_source: 'record' to use bag record time (t), or 'header' to use
                     message header stamps when available.
    """
    if time_source not in ("record", "header"):
        raise ValueError("time_source must be 'record' or 'header'")

    if time_source == "record":
        # Use the bag record time directly; this avoids cross-topic clock skew.
        messages = list(iterate_bag(bag))
        messages.sort(key=lambda x: x[2])
        for topic, msg, t in messages:
            yield topic, msg, t
        return

    # Header-based ordering (legacy). This can fail if topics use different clocks.
    topic_types = {topic.name: topic.type for topic in bag.get_all_topics_and_types()}

    def _get_header_timestamp(topic, msg, t):
        msg_type = get_message(topic_types[topic])
        msg = deserialize_message(msg, msg_type)
        if hasattr(msg, "header"):
            return int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        return t

    unsorted_messages = [
        (topic, msg, _get_header_timestamp(topic, msg, t))
        for topic, msg, t in iterate_bag(bag)
    ]
    sorted_messages = sorted(unsorted_messages, key=lambda x: x[2])
    for topic, msg, t in sorted_messages:
        yield topic, msg, t


def process_frame(frame: dict) -> dict:
    frame["state"] = np.array(parse_droid_joint_state(frame["state"]), dtype=np.float32)

    actions = frame.pop("actions")
    final_joint_state, _ = actions[-1]
    frame["action"] = np.array(parse_droid_joint_state(final_joint_state), dtype=np.float32)

    return frame


def load_bag(bag_file_path: str, dataset: LeRobotDataset, task_description: str, *, time_source: str = "record"):
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_file_path, storage_id="sqlite3")
    converter_options = ConverterOptions()
    reader.open(storage_options, converter_options)

    frame = {
        "image": None,
        "wrist_image": None,
        "state": None,
        "task": task_description,
        "actions": [],
    }

    prev_state = None
    joint_reordering = None

    frame_count = 0

    first_timestamp = None
    last_timestamp = None

    # debug counters
    total_messages = 0
    image_msgs = 0
    wrist_msgs = 0
    joint_msgs = 0
    actions_appended = 0

    def explain_frame_state(f):
        reasons = []
        if f["image"] is None:
            reasons.append("missing image")
        if f["wrist_image"] is None:
            reasons.append("missing wrist_image")
        if f["state"] is None:
            reasons.append("missing state")
        if len(f["actions"]) < 1:
            reasons.append("no actions")
        return ", ".join(reasons) if reasons else "valid"

    for topic, msg, t in reorder_bag_timestamps(reader, time_source=time_source):
        total_messages += 1
        if first_timestamp is None:
            first_timestamp = t
        # keep track of the most recent timestamp
        last_timestamp = t

        # If the arrival of an image should trigger adding the previous frame, check validity first
        if frame_is_valid(frame) and (
            topic == "/wrist_camera/image_raw"
            or topic == "/zed/zed_node/right/image_rect_color"
        ):
            dataset.add_frame(process_frame(frame))
            frame_count += 1
            logger.info(
                "Added frame #%d from bag %s at topic=%s timestamp=%s",
                frame_count,
                bag_file_path,
                topic,
                t,
            )

            # Start a new frame
            frame = {
                "image": None,
                "wrist_image": None,
                "state": None,
                "task": task_description,
                "actions": [],
            }

        if topic == "/wrist_camera/image_raw":
            wrist_msgs += 1
            if frame["wrist_image"] is None:
                frame["wrist_image"] = parse_image(msg)
                logger.debug("Set wrist_image for current frame (bag=%s time=%s)", bag_file_path, t)
                if frame["state"] is None:
                    frame["state"] = prev_state
                    logger.debug("Initialized frame state from prev_state after wrist image (bag=%s)", bag_file_path)

        elif topic == "/zed/zed_node/right/image_rect_color":
            image_msgs += 1
            if frame["image"] is None:
                frame["image"] = parse_image(msg)
                logger.debug("Set image for current frame (bag=%s time=%s)", bag_file_path, t)
                if frame["state"] is None:
                    frame["state"] = prev_state
                    logger.debug("Initialized frame state from prev_state after image (bag=%s)", bag_file_path)
        elif topic == "/joint_states":
            joint_msgs += 1
            if joint_reordering is None:
                prev_state = msg
                _msg = deserialize_message(msg, JointState)
                joint_reordering = [
                    _msg.name.index("panda_joint1"),
                    _msg.name.index("panda_joint2"),
                    _msg.name.index("panda_joint3"),
                    _msg.name.index("panda_joint4"),
                    _msg.name.index("panda_joint5"),
                    _msg.name.index("panda_joint6"),
                    _msg.name.index("panda_joint7"),
                    _msg.name.index("finger_joint"),
                ]
                logger.info("Computed joint_reordering=%s names=%s (bag=%s)", joint_reordering, _msg.name, bag_file_path)

            if frame["state"] is not None:
                frame["actions"].append((msg, t))
                actions_appended += 1
                logger.debug("Appended action (bag=%s time=%s). total actions for frame=%d", bag_file_path, t, len(frame["actions"]))
                prev_state = msg
            elif frame["image"] is not None and frame["wrist_image"] is not None:
                frame["state"] = msg
                logger.debug("Set frame state from joint_states (bag=%s time=%s)", bag_file_path, t)

        else:
            # print(f"Topic: {topic}, Time: {t}")
            pass

        # If this message is an image-trigger and frame wasn't valid, explain why we didn't add it
        if topic in ("/wrist_camera/image_raw", "/zed/zed_node/right/image_rect_color") and not frame_is_valid(frame):
            reason = explain_frame_state(frame)
            # logger.info(
            #     "Skipping add_frame at trigger topic=%s time=%s for bag=%s — reasons: %s",
            #     topic,
            #     t,
            #     bag_file_path,
            #     reason,
            # )

    # Only save an episode if we actually added frames
    if frame_count == 0:
        logger.warning(
            "No frames extracted from bag %s — summary: total_messages=%d image_msgs=%d wrist_msgs=%d joint_msgs=%d actions_appended=%d",
            bag_file_path,
            total_messages,
            image_msgs,
            wrist_msgs,
            joint_msgs,
            actions_appended,
        )
        return

    dataset.save_episode()

    if first_timestamp is not None and last_timestamp is not None:
        # Convert ns->s using integer division to avoid float rounding on huge stamps.
        start_dt = datetime.datetime.fromtimestamp(int(first_timestamp) // 1_000_000_000)
        end_dt = datetime.datetime.fromtimestamp(int(last_timestamp) // 1_000_000_000)
        episode_time = end_dt - start_dt
        print(f"Time range: {episode_time}, frames: {frame_count}")
        # Warn if time range is suspiciously large (likely mixed clocks when using header time).
        if episode_time.days > 1_000 and time_source == "header":
            logger.warning(
                "Very large episode time range detected (%s). Consider --time-source=record to avoid mixed topic clocks.",
                episode_time,
            )
    else:
        print(f"Frames: {frame_count} (timestamps unavailable)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_bag_parser = subparsers.add_parser("single", help="Process a single bag")
    single_bag_parser.add_argument(
        "--bag_path",
        type=str,
        required=True,
        help="Path to the single bag file to process",
    )
    single_bag_parser.add_argument(
        "--task_description",
        type=str,
        required=True,
        help="Task description for the single bag",
    )
    single_bag_parser.add_argument("--dataset_name", type=str, required=True)


    csv_parser = subparsers.add_parser("csv", help="Process all bags in a CSV")
    csv_parser.add_argument(
        "--tasks_csv",
        type=str,
        help="Path to the CSV file containing the list of bags to process",
    )
    csv_parser.add_argument(
        "--bags_dir", type=str, required=True, help="Directory containing the bag files"
    )
    csv_parser.add_argument("--dataset_name", type=str, required=True)
    csv_parser.add_argument(
        "--time_source",
        type=str,
        choices=["record", "header"],
        default="record",
        help="Use 'record' (bag record time) or 'header' (message header stamp) to order messages",
    )

    # parse args after subparsers and args have been defined
    args = parser.parse_args()

    if args.mode == "single":
        bags = [args.bag_path]
        task_descriptions = [args.task_description]
    else:
        tasks = pd.read_csv(args.tasks_csv)
        bags = [
            os.path.join(args.bags_dir, row["filename"]) for _, row in tasks.iterrows()
        ]
        task_descriptions = [row["task_description"] for _, row in tasks.iterrows()]

    dataset = LeRobotDataset.create(
        args.dataset_name,
        fps=15,
        features={
            "image": {
                "dtype": "image",
                # "dtype": "video",
                "shape": (3, 224, 224),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            },
            "wrist_image": {
                "dtype": "image",
                # "dtype": "video",
                "shape": (3, 224, 224),
                "names": [
                    "channels",
                    "height",
                    "width",
                ],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["action"],
            },
        },
        image_writer_threads=4,
    )

    for bag_file_path, task_description in zip(bags, task_descriptions):
        load_bag(bag_file_path, dataset, task_description, time_source=getattr(args, "time_source", "record"))

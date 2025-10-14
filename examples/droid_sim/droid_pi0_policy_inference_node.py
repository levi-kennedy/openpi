#!/usr/bin/env python3
import logging
import pathlib
import dataclasses
import queue
import threading
from typing_extensions import override, Protocol, TypeAlias
import numpy as np
import time
import einops
import tyro
import jax

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge

# OpenPI imports
from openpi.models import model as _model
from openpi.models import pi0
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory, AssetsConfig
from openpi.training import weight_loaders
from openpi import transforms as _transforms
#from openpi.policies.libero_policy import _parse_image
from openpi.training.config import TrainConfig
import openpi.models.pi0_fast as pi0_fast

ModelType: TypeAlias = _model.ModelType
DEFAULT_RANDOM_SEED = 0

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(_transforms.DataTransformFn):
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0_FAST

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["observation/joint_position"], data["observation/gripper_position"]])
        state = _transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=lambda: (lambda model_config: _transforms.Group(inputs=[], outputs=[]))
    )
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=ModelTransformFactory
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )




class DroidPolicyNode(Node):
    def __init__(self):
        super().__init__('droid_policy_node')

        logging.basicConfig(level=logging.INFO, force=True)
        self.get_logger().info("Loading Droid Pi0 policy...")

        # Debugging: save one observation to disk for offline inspection
        self._debug_obs_saved = False  # Only save once
        self._debug_obs_path = None  # pathlib.Path("/tmp/droid_pi0_obs.npz")

        # local fin-tuned lora
        checkpoint_dir = pathlib.Path("/home/levi/projects/openpi/checkpoints/pi0_fast_droid_local_lora/droid-marker-bowl1/29999")
        model_cfg = pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora")
        asset_id = "droid_marker_in_bowl"
        #base
        # checkpoint_dir = pathlib.Path("/home/levi/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base")
        # model_cfg = pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10)
        # pi0 FAST droid checkpoint
        # checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
        # model_cfg = pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10)      
        # asset_id = "droid"

        # Build train config and policy using DROID transforms.
        config = TrainConfig(
            name="pi0_droid_fast",
            model=model_cfg,
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id=asset_id),
                data_transforms=lambda model: _transforms.Group(
                    inputs=[
                        # _transforms.DeltaActions(delta_mask),
                        DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST),
                    ],
                    outputs=[
                        # _transforms.AbsoluteActions(delta_mask),
                        DroidOutputs(),
                    ],
                ),
                base_config=DataConfig(
                    prompt_from_task=True,
                ),
            ),
        )        

        # Create the pi0 policy.
        self.policy = _policy_config.create_trained_policy(
            train_config=config,
            checkpoint_dir=checkpoint_dir,
        )
        self.get_logger().info("Policy loaded.")

        # Seed both NumPy and JAX policy RNG for reproducibility.
        self._seed = DEFAULT_RANDOM_SEED
        np.random.seed(self._seed)
        self.policy._rng = jax.random.PRNGKey(self._seed)

        self.cv_bridge = CvBridge()
        self._state_lock = threading.Lock()
        self.latest_joint_state = None
        self.latest_base_image = None
        self.latest_wrist_image = None
        self.processing = False  # Guard for overlapping inference calls, protected by self._state_lock

        # ROS2 subscriptions and publications for robot state and images
        # Use reliable QoS for important topics
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
        # Use a reentrant callback group to allow concurrent callbacks, otherwise image, joint state and inference callbacks can block each other
        self.callback_group = ReentrantCallbackGroup()
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos, callback_group=self.callback_group)
        self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.base_image_cb, qos, callback_group=self.callback_group)
        self.create_subscription(Image, '/wrist_camera/image_raw', self.wrist_image_cb, qos, callback_group=self.callback_group)
        # Guard condition to trigger inference outside of callbacks (basically a manually triggered event)
        self._inference_guard = self.create_guard_condition(self._on_inference_trigger, callback_group=self.callback_group)

        self.robot_action_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', qos)
        self.trajectory_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "finger_joint"  # Gripper joint
        ]

        # Background action publisher keeps streaming velocities while next inference runs
        self._action_queue = queue.Queue(maxsize=1)
        self._publisher_shutdown = threading.Event()
        self._publisher_thread = threading.Thread(target=self._action_publisher_loop, name="action_publisher", daemon=True)
        self._publisher_thread.start()  # start the publisher thread immediately (sits waiting for actions)

        self._first_inference_requested = False

        
        

    # Callbacks
    def joint_state_cb(self, msg: JointState):
        with self._state_lock:
            self.latest_joint_state = np.array(msg.position, dtype=np.float32)
            self.trajectory_joint_names = msg.name[:8] # Assuming the first 8 joints are the robot joints
        self._maybe_request_inference()
        #self.get_logger().info(f"Joint state updated: {self.latest_joint_state}")


    def base_image_cb(self, msg: Image):
        try:
            converted = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self._state_lock:
                self.latest_base_image = converted
            self._maybe_request_inference()
            #self.get_logger().info("Base image updated.")
        except Exception as e:
            self.get_logger().error(f"Base image conversion error: {e}")

    def wrist_image_cb(self, msg: Image):
        try:
            converted = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            with self._state_lock:
                self.latest_wrist_image = converted
            self._maybe_request_inference()
            #self.get_logger().info("Wrist image updated.")
        except Exception as e:
            self.get_logger().error(f"Wrist image conversion error: {e}")

    
    def process_and_publish(self):
        # Prevent multiple concurrent executions and copy state under lock
        with self._state_lock:
            # If already processing, skip this call to avoid overlap
            if self.processing:
                return
            self.processing = True

            data_missing = (
                self.latest_joint_state is None
                or self.latest_base_image is None
                or self.latest_wrist_image is None
            )

            if data_missing:
                self.processing = False
                self.get_logger().info("Waiting for sensor data...", throttle_duration_sec=5.0)
                return
            
            

        try:
            # Create observation dictionary
            
            obs = {
                "observation/exterior_image_1_left": self.latest_base_image,
                "observation/wrist_image_left": self.latest_wrist_image,
                "observation/joint_position": self.latest_joint_state[:7],  # First 7 joints for the panda arm
                "observation/gripper_position": np.array([self.latest_joint_state[7]]), # Assuming the 2f-85 gripper is the last joint
                "prompt": "Pick up the marker from the table and put it in the bowl",
            }

            # Save observation snapshot for offline debugging if enabled
            if self._debug_obs_path is not None and not self._debug_obs_saved:
                self._dump_observation_snapshot(obs)

            # Run inference
            self.get_logger().info("Running policy inference...")
            # compute time required for inference
            start_time = time.time()
            result = self.policy.infer(obs)

            if result is None or 'actions' not in result:
                self.get_logger().warn("No actions returned from policy.")
            else:
                actions = result['actions'] 
                inference_time = time.time() - start_time
                self.get_logger().info(f"Policy inference complete in {inference_time:.4f} seconds ... sending actions to queue")
                if self._enqueue_actions(actions):  # dump the actions into the queue for the publisher thread
                    self.get_logger().info("Action queued for publishing.")
        except Exception as e:
            self.get_logger().error(f"Error in processing loop: {e}")
        finally:
            with self._state_lock:
                self.processing = False

    def _dump_observation_snapshot(self, obs: dict) -> None:
        """Save the current observation dictionary to disk for offline debug inspection."""
        try:
            save_data: dict[str, np.ndarray] = {}
            for key, value in obs.items():
                sanitized_key = key.replace("/", "__")
                if isinstance(value, np.ndarray):
                    save_data[sanitized_key] = value.copy()
                else:
                    save_data[sanitized_key] = np.asarray([value])
            np.savez_compressed(self._debug_obs_path, **save_data)
            self.get_logger().info(f"Saved observation snapshot to {self._debug_obs_path}")
        except Exception as exc:  
            self.get_logger().error(f"Failed to save observation snapshot: {exc}")
        finally:
            self._debug_obs_saved = True

    def _enqueue_actions(self, actions: np.ndarray) -> bool:
        actions = np.asarray(actions)
        try:
            self._action_queue.put_nowait(actions)
            return True
        except queue.Full:
            try:
                self._action_queue.get_nowait()
                self._action_queue.task_done()
                self.get_logger().warn("Replacing queued action sequence with newer inference result.")
            except queue.Empty:
                pass
            try:
                self._action_queue.put_nowait(actions)
                return True
            except queue.Full:
                self.get_logger().warn("Action queue full; dropping newly generated sequence.")
                return False

    def _request_inference(self) -> None:
        if self._publisher_shutdown.is_set():
            return
        self._inference_guard.trigger()

    def _maybe_request_inference(self) -> None:
        # Check if we have all required data and whether this is the first time inference has been requested
        # We only want to call inference here to kick off the process once, subsequent inferences are triggered
        # by the action publisher thread when it finishes sending the current action sequence
        with self._state_lock:
            ready = (
                self.latest_joint_state is not None
                and self.latest_base_image is not None
                and self.latest_wrist_image is not None
                and not self.processing
            )            

            if self._first_inference_requested or not ready:
                return
            self._request_inference()
            self._first_inference_requested = True

    def _on_inference_trigger(self) -> None:
        with self._state_lock:
            if self.processing:
                return
        self.process_and_publish()

    def _action_publisher_loop(self) -> None:
        while not self._publisher_shutdown.is_set():
            try:
                actions = self._action_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._publish_action_sequence(actions)
            except Exception as exc:  
                self.get_logger().error(f"Failed to publish action sequence: {exc}")
            finally:
                self._action_queue.task_done()

    def _publish_action_sequence(self, actions: np.ndarray) -> None:
        with self._state_lock:
            joint_names = list(self.trajectory_joint_names)
            self._publishing_actions = True

        total_actions = len(actions)

        try:
            for idx, action in enumerate(actions):
                if self._publisher_shutdown.is_set():
                    break
                
                robot_traj = JointTrajectory()
                robot_traj.joint_names = joint_names
                robot_traj.header.stamp = self.get_clock().now().to_msg()
                robot_traj.header.frame_id = "base_link"
                joint_pt = JointTrajectoryPoint()
                positions = np.zeros(8, dtype=np.float32)
                positions[7] = action[7] * 1.5  # Scale gripper position command
                joint_pt.positions = positions.clip(0.0, 1.0).tolist()
                velocities = np.zeros(8, dtype=np.float32)
                velocities[:7] = action[:7]  * 1.0  # Scale velocities if you want to slow down the action rate
                joint_pt.velocities = velocities.tolist()
                joint_pt.time_from_start.nanosec = 1_000_000  # 0.1 sec, doesn't affect anything
                robot_traj.points = [joint_pt]
                self.robot_action_pub.publish(robot_traj)   

                print(
                    f"Action Pub{idx+1}/{total_actions}: gripper pos {np.array2string(np.array([positions[7]]), precision=1, suppress_small=True, floatmode='fixed')}, "
                    f"arm vel {np.array2string(velocities[:7], precision=3, suppress_small=True, floatmode='fixed')}"
                )
                
                if idx == total_actions - 1:
                    self.get_logger().info("Requesting next inference run.")
                    self._request_inference()

                    # send Trajectory message with zero velocities to stop the robot
                    # stop_traj = JointTrajectory()
                    # stop_traj.joint_names = joint_names
                    # stop_traj.header.stamp = self.get_clock().now().to_msg()
                    # stop_traj.header.frame_id = "base_link"
                    # stop_pt = JointTrajectoryPoint()
                    # stop_pt.positions = joint_pt.positions
                    # stop_pt.velocities = [0.0] * 8
                    # stop_traj.points = [stop_pt]
                    # self.robot_action_pub.publish(stop_traj)
                    # self.get_logger().info("Published stop command to robot.")

                time.sleep(0.1)  # Sleep to maintain 15Hz command rate


        finally:
            with self._state_lock:
                self._publishing_actions = False

    def destroy_node(self):
        self._publisher_shutdown.set()
        if self._publisher_thread.is_alive():
            self._publisher_thread.join(timeout=1.0)
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = DroidPolicyNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

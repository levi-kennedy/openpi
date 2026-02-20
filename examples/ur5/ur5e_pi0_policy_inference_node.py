#!/usr/bin/env python3
import argparse
import os
import logging
import dataclasses
import queue
import threading
from typing_extensions import override, Protocol, TypeAlias
import numpy as np
import time
import einops
import jax

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.utilities import remove_ros_args
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from cv_bridge import CvBridge

# OpenPI imports
from openpi.policies import policy_config as _policy_config
from openpi import transforms as _transforms
from openpi.training import config as train_config


DEFAULT_RANDOM_SEED = 0
DEFAULT_PROMPT = "Pick up the marker from the table and put it in the bowl"


def _create_policy(config_name: str, checkpoint_dir: str):
    cfg = train_config.get_config(config_name)
    resolved_checkpoint_dir = os.path.abspath(os.path.expanduser(checkpoint_dir))
    policy = _policy_config.create_trained_policy(
        train_config=cfg,
        checkpoint_dir=resolved_checkpoint_dir,
        # This maps ros2 topic/input names to the expected policy input names.
        repack_transforms=_transforms.Group(inputs=[
            _transforms.RepackTransform({
                "image": "base_rgb",
                "wrist_image": "wrist_rgb",
                "joints": "joints",
                "gripper": "gripper",
                "prompt": "prompt",
            })
        ]),
    )
    return policy, resolved_checkpoint_dir


def _seed_policy(policy, seed: int) -> None:
    np.random.seed(seed)
    policy._rng = jax.random.PRNGKey(seed)


class UR5ePolicyNode(Node):
    def __init__(self, policy):
        super().__init__('ur5e_policy_node')

        logging.basicConfig(level=logging.INFO, force=True)
        self.policy = policy

        self.declare_parameter("prompt", DEFAULT_PROMPT)

        # Debugging: save one observation to disk for offline inspection
        self._debug_obs_saved = False  # Only save once
        self._debug_obs_path = None
        
        self.get_logger().info("Policy loaded.")

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
        self.create_subscription(Image, '/base_camera/zed_node/rgb/color/rect/image', self.base_image_cb, qos, callback_group=self.callback_group)
        self.create_subscription(Image, '/wrist_camera/zed_node/rgb/color/rect/image', self.wrist_image_cb, qos, callback_group=self.callback_group)
        # Guard condition to trigger inference outside of callbacks (basically a manually triggered event)
        self._inference_guard = self.create_guard_condition(self._on_inference_trigger, callback_group=self.callback_group)

        self.robot_action_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', qos)
        self.trajectory_joint_names = [
            "ur5e_shoulder_pan_joint",
            "ur5e_shoulder_lift_joint",
            "ur5e_elbow_joint",
            "ur5e_wrist_1_joint",
            "ur5e_wrist_2_joint",
            "ur5e_wrist_3_joint",
        ]

        # Gripper control client
        self.gripper_action_client = ActionClient(
            self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd"
        )

        # Background action publisher keeps streaming velocities while next inference runs
        self._action_queue = queue.Queue(maxsize=1)
        self._publisher_shutdown = threading.Event()
        self._publisher_thread = threading.Thread(target=self._action_publisher_loop, name="action_publisher", daemon=True)
        self._publisher_thread.start()  # start the publisher thread immediately (sits waiting for actions)

        self._first_inference_requested = False

        
        

    # Callbacks
    def joint_state_cb(self, msg: JointState):
        with self._state_lock:
            current_joint_state = [0] * 7
            for i, joint_name in enumerate(self.trajectory_joint_names):
                idx = msg.name.index(joint_name)
                current_joint_state[i] = msg.position[idx]
            # Hardcode the position of the gripper without having it's name in the list of trajectory_joint_names
            current_joint_state[-1] = msg.position[0]
            self.latest_joint_state = np.array(current_joint_state, dtype=np.float32)
            # self.trajectory_joint_names = msg.name[:7]  # Use the six arm joints + gripper
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
                "base_rgb": self.latest_base_image.copy(),
                "wrist_rgb": self.latest_wrist_image.copy(),
                "joints": self.latest_joint_state[:6].copy(),  # Ur5e has 6 arm joints
                "gripper": np.array([self.latest_joint_state[6]], dtype=np.float32),  # 2f-85 gripper last joint
                "prompt": str(self.get_parameter("prompt").value),
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
                positions = np.zeros(7, dtype=np.float32)
                positions = action[:6]
                joint_pt.positions = positions.tolist()
                joint_pt.time_from_start.nanosec = 1_000_000  # 0.1 sec
                robot_traj.points = [joint_pt]
                self.robot_action_pub.publish(robot_traj)

                def noop(*args):
                    pass
                goal_msg = GripperCommand.Goal()
                gripper_position = action[6]
                goal_msg.command.position = gripper_position
                goal_msg.command.max_effort = 10.0

                send_goal_future = self.gripper_action_client.send_goal_async(
                    goal_msg, feedback_callback=noop
                )
                send_goal_future.add_done_callback(noop)

                print(
                    f"Action Pub{idx+1}/{total_actions}: "
                    f"arm pos {np.array2string(positions, precision=3, suppress_small=True, floatmode='fixed')}"
                )
                if idx == total_actions - 1:
                    self.get_logger().info("Requesting next inference run.")
                    self._request_inference()

                time.sleep(0.09)


        finally:
            with self._state_lock:
                self._publishing_actions = False

    def destroy_node(self):
        self._publisher_shutdown.set()
        if self._publisher_thread.is_alive():
            self._publisher_thread.join(timeout=1.0)
        return super().destroy_node()

def main(args=None):
    parser = argparse.ArgumentParser(
        description="UR5e Pi0 policy inference node."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Config name passed to openpi.training.config.get_config().",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the trained policy checkpoint directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used for policy RNG initialization.",
    )
    cli_args = remove_ros_args(args=args)
    parsed = parser.parse_args(cli_args[1:])

    policy, checkpoint_dir = _create_policy(
        config_name=parsed.config,
        checkpoint_dir=parsed.checkpoint_dir,
    )
    _seed_policy(policy, parsed.seed)

    rclpy.init(args=args)
    node = UR5ePolicyNode(
        policy=policy,
    )
    node.get_logger().info(
        f"Using config '{parsed.config}' and checkpoint '{checkpoint_dir}'."
    )
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

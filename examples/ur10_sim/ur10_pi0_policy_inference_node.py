#!/usr/bin/env python3
import logging
import pathlib
import dataclasses
from typing_extensions import override
import numpy as np
import cv2

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy
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
from openpi.policies.libero_policy import _parse_image
from openpi.training.config import TrainConfig

# --- Data transforms for UR10 ---
@dataclasses.dataclass(frozen=True)
class UR10Inputs(_transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0
        # Concatenate joints and gripper
        state = np.concatenate([data["joints"], data["gripper"]])
        state = _transforms.pad_to_dim(state, self.action_dim)
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = _transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs

@dataclasses.dataclass(frozen=True)
class UR10Outputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}

@dataclasses.dataclass(frozen=True)
class LeRobotUR10DataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack = _transforms.Group(inputs=[
            _transforms.RepackTransform({
                "base_rgb": "image",
                "wrist_rgb": "wrist_image",
                "joints": "joints",
                "gripper": "gripper",
                "prompt": "prompt",
            })
        ])
        data_transforms = _transforms.Group(
            inputs=[UR10Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR10Outputs()],
        )
        delta_mask = _transforms.make_bool_mask(6, -1)        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_mask)],
            outputs=[_transforms.AbsoluteActions(delta_mask)],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

class UR10PolicyNode(Node):
    def __init__(self):
        super().__init__('ur10_policy_node')

        logging.basicConfig(level=logging.INFO)
        self.get_logger().info("Loading UR10 Pi0 policy...")

        # Prepare data/config for policy
        data_factory = LeRobotUR10DataConfig(
            repo_id=None,
            assets=AssetsConfig(
                assets_dir="/home/levi/projects/openpi/examples",
                asset_id="UR10",
            ),
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        )
        config = TrainConfig(
            name="pi0_ur10",
            model=pi0.Pi0Config(),
            data=data_factory,
            weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_droid/params"),
            num_train_steps=30000,
        )
        data_cfg = config.data.create(config.assets_dirs, config.model)
        norm_stats = data_cfg.norm_stats
        if norm_stats is None:
            raise ValueError(
                f"Normalization stats failed to load from {config.data.assets.assets_dir}/{config.data.assets.asset_id}/norm_stats.json"
            )
        checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_droid")
        self.policy = _policy_config.create_trained_policy(
            train_config=config,
            checkpoint_dir=checkpoint_dir,
            norm_stats=norm_stats,
        )
        self.get_logger().info("Policy loaded.")
        # CV Bridge
        self.cv_bridge = CvBridge()
        # State holders
        self.latest_joint_state = None
        self.latest_base_image = None
        self.latest_wrist_image = None
        # QoS
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE)
        # Subscribers
        self.create_subscription(JointState, '/joint_states', self.joint_state_cb, qos)
        self.create_subscription(Image, '/zed/zed_node/left/image_rect_color', self.base_image_cb, qos)
        self.create_subscription(Image, '/wrist_camera/image_raw', self.wrist_image_cb, qos)
        # Publisher
        self.action_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', qos)
        self.trajectory_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    # Callbacks
    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = np.array(msg.position, dtype=np.float32)

    def base_image_cb(self, msg: Image):
        try:
            self.latest_base_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Base image conversion error: {e}")

    def wrist_image_cb(self, msg: Image):
        try:
            self.latest_wrist_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f"Wrist image conversion error: {e}")

    def run_control_loop(self):
        rate = self.create_rate(10)
        while rclpy.ok():
            # Process callbacks
            rclpy.spin_once(self, timeout_sec=0.0)
            if (self.latest_joint_state is None or
                self.latest_base_image is None or
                self.latest_wrist_image is None):
                self.get_logger().info("Waiting for observation data...", throttle_duration_sec=5)
                import time; time.sleep(0.5)
                continue
            # Build obs
            joints = self.latest_joint_state.copy()
            gripper = np.array([joints[6]])
            obs = {
                "base_rgb": self.latest_base_image.copy(),
                "wrist_rgb": self.latest_wrist_image.copy(),
                "joints": joints[:6],
                "gripper": gripper,
                "prompt": "touch the yellow block with the robot gripper",
            }
            # Perform inference on the observation
            self.get_logger().info("Running policy inference...")
            result = self.policy.infer(obs)

            # Now apply the result action to the robot by publishing a JointTrajectory
            if result is None or 'actions' not in result:
                self.get_logger().warn("No actions returned from policy.")
            else:
                actions = result['actions']
                for i in range(min(len(actions), 50)):
                    traj = JointTrajectory()
                    traj.joint_names = self.trajectory_joint_names
                    traj.header.stamp = self.get_clock().now().to_msg()
                    traj.header.frame_id = "base_link"
                    pt = JointTrajectoryPoint()
                    pt.positions = actions[i, :6].tolist()
                    traj.points = [pt]
                    self.action_pub.publish(traj)
                    import time; time.sleep(0.1)
                self.get_logger().info("Published action sequence.")
            # Reset state
            self.latest_joint_state = None
            self.latest_base_image = None
            self.latest_wrist_image = None
            try:
                rate.sleep()
            except rclpy.executors.ExternalShutdownException:
                break


def main(args=None):
    rclpy.init(args=args)
    node = UR10PolicyNode()
    try:
        node.run_control_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

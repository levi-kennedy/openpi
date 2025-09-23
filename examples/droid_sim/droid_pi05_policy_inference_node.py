#!/usr/bin/env python3
import dataclasses
import logging
import pathlib
import time
from collections.abc import Callable

import einops
import numpy as np
from typing_extensions import Protocol, TypeAlias, override

# ROS2 imports
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# OpenPI imports
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
    TrainConfig,
)

ModelType: TypeAlias = _model.ModelType
DEFAULT_VARIANT = "pi05"
DROID_ACTION_DIM = 8


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(_transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state = np.concatenate(
            [data["observation/joint_position"], data["observation/gripper_position"]]
        )
        state = _transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
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
        return {"actions": np.asarray(data["actions"][:, :DROID_ACTION_DIM])}


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        ...


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    data_transforms: GroupFactory = dataclasses.field(
        default_factory=lambda: (lambda model_config: _transforms.Group(inputs=[], outputs=[]))
    )
    model_transforms: GroupFactory = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class PolicyVariant:
    model_type: ModelType
    config_factory: Callable[[], _model.BaseModelConfig]
    default_checkpoint: str
    train_config_name: str
    action_publish_dim: int = DROID_ACTION_DIM


_POLICY_VARIANTS: dict[str, PolicyVariant] = {
    "pi05": PolicyVariant(
        model_type=ModelType.PI05,
        config_factory=lambda: pi0_config.Pi0Config(action_horizon=15, pi05=True),
        default_checkpoint="gs://openpi-assets/checkpoints/pi05_droid",
        train_config_name="pi05_droid",
    ),
    "pi05_fast": PolicyVariant(
        model_type=ModelType.PI0_FAST,
        config_factory=lambda: pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        default_checkpoint="gs://openpi-assets/checkpoints/pi05_fast_droid",
        train_config_name="pi05_fast_droid",
    ),
}


class DroidPolicyNode(Node):
    def __init__(self):
        super().__init__("droid_policy_node")

        logging.basicConfig(level=logging.INFO, force=True)

        self.declare_parameter("policy.model_variant", DEFAULT_VARIANT)
        variant_key = (
            self.get_parameter("policy.model_variant").get_parameter_value().string_value or DEFAULT_VARIANT
        ).lower()
        variant = _POLICY_VARIANTS.get(variant_key)
        if variant is None:
            self.get_logger().warning(
                "Unknown model variant '%s'; falling back to '%s'.", variant_key, DEFAULT_VARIANT
            )
            variant_key = DEFAULT_VARIANT
            variant = _POLICY_VARIANTS[variant_key]

        self.declare_parameter("policy.checkpoint", "")
        checkpoint_override = (
            self.get_parameter("policy.checkpoint").get_parameter_value().string_value
        ).strip()

        checkpoint_path = checkpoint_override or variant.default_checkpoint
        if not checkpoint_path:
            raise ValueError(
                "No checkpoint path provided. Set 'policy.checkpoint' parameter to a valid checkpoint URI."
            )

        self.get_logger().info(
            f"Loading DROID policy (variant={variant_key}) from checkpoint {checkpoint_path}..."
        )

        model_cfg = variant.config_factory()
        checkpoint_dir = download.maybe_download(checkpoint_path)

        delta_mask = np.array([1] * 7 + [0] * 1, dtype=bool) # which dimensions are deltas vs absolutes, all but gripper
        data_transforms = lambda model: _transforms.Group(  # noqa: E731
            inputs=[DroidInputs(action_dim=model.action_dim, model_type=variant.model_type)],
            outputs=[
                _transforms.AbsoluteActions(delta_mask),
                DroidOutputs()
            ],
        )

        train_config = TrainConfig(
            name=f"{variant.train_config_name}_inference",
            model=model_cfg,
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
                data_transforms=data_transforms,
                base_config=DataConfig(prompt_from_task=True),
            ),
        )

        self.policy = _policy_config.create_trained_policy(
            train_config=train_config,
            checkpoint_dir=checkpoint_dir,
        )
        self.get_logger().info("Policy loaded.")

        self._variant = variant
        self._action_publish_dim = variant.action_publish_dim
        self._action_horizon = model_cfg.action_horizon

        self.cv_bridge = CvBridge()
        self.latest_joint_state: np.ndarray | None = None
        self.latest_base_image: np.ndarray | None = None
        self.latest_wrist_image: np.ndarray | None = None

        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, qos)
        self.create_subscription(Image, "/zed/zed_node/left/image_rect_color", self.base_image_cb, qos)
        self.create_subscription(Image, "/wrist_camera/image_raw", self.wrist_image_cb, qos)

        self.timer = self.create_timer(10.0, self.process_and_publish)
        self.processing = False

        self.robot_action_pub = self.create_publisher(
            JointTrajectory,
            "/scaled_joint_trajectory_controller/joint_trajectory",
            qos,
        )
        self.trajectory_joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "finger_joint",
        ]

        self.debug_data = np.zeros((100, DROID_ACTION_DIM, 3), dtype=np.float32)
        self.didx = 0

        self._debug_obs_saved = False
        self._debug_obs_path = pathlib.Path(
            "/home/levi/projects/openpi/examples/droid_sim/debug_obs.npz"
        )
        self._debug_obs_path.parent.mkdir(parents=True, exist_ok=True)

    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = np.array(msg.position, dtype=np.float32)
        self.trajectory_joint_names = msg.name[:DROID_ACTION_DIM]

    def base_image_cb(self, msg: Image):
        try:
            self.latest_base_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:  # pragma: no cover - ROS callback
            self.get_logger().error(f"Base image conversion error: {exc}")

    def wrist_image_cb(self, msg: Image):
        try:
            self.latest_wrist_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception as exc:  # pragma: no cover - ROS callback
            self.get_logger().error(f"Wrist image conversion error: {exc}")

    def process_and_publish(self):
        if self.processing:
            return
        self.processing = True

        if (
            self.latest_joint_state is None
            or self.latest_base_image is None
            or self.latest_wrist_image is None
        ):
            throttle_duration_sec = 5.0
            self.get_logger().info(f"Waiting for sensor data...")
            self.processing = False
            return

        try:
            robot_joints = self.latest_joint_state.copy()
            robot = robot_joints[:7]
            gripper = np.array([robot_joints[7]])
            obs = {
                "observation/exterior_image_1_left": self.latest_base_image.copy(),
                "observation/wrist_image_left": self.latest_wrist_image.copy(),
                "observation/joint_position": robot,
                "observation/gripper_position": gripper,
                "prompt": "Touch the blue coffee cup with the robot's gripper.",
            }

            if not self._debug_obs_saved and self._debug_obs_path is not None:
                self._dump_observation_snapshot(obs)

            self.get_logger().info("Running policy inference...")
            result = self.policy.infer(obs)

            if not result or "actions" not in result:
                self.get_logger().warning("No actions returned from policy.")
            else:
                actions = np.asarray(result["actions"])
                publish_dim = min(self._action_publish_dim, actions.shape[1])
                publish_horizon = min(self._action_horizon, actions.shape[0])
                for i in range(publish_horizon):
                    joint_traj = JointTrajectory()
                    joint_traj.joint_names = self.trajectory_joint_names
                    joint_traj.header.stamp = self.get_clock().now().to_msg()
                    joint_traj.header.frame_id = "base_link"

                    joint_pt = JointTrajectoryPoint()
                    joint_pt.positions = actions[i, :publish_dim].tolist()
                    joint_traj.points = [joint_pt]
                    self.robot_action_pub.publish(joint_traj)

                    self.debug_data[self.didx, :publish_dim, 0] = actions[i, :publish_dim]
                    time.sleep(0.1)
                self.get_logger().info("Published action sequence.")

            if self.didx >= self.debug_data.shape[0]:
                self.didx = 0
        except Exception as exc:  # pragma: no cover - ROS callback
            self.get_logger().error(f"Error in processing loop: {exc}")
        finally:
            self.processing = False

    def _dump_observation_snapshot(self, obs: dict) -> None:
        try:
            payload: dict[str, np.ndarray] = {}
            for key, value in obs.items():
                sanitized_key = key.replace("/", "__")
                if isinstance(value, np.ndarray):
                    payload[sanitized_key] = value.copy()
                else:
                    payload[sanitized_key] = np.asarray([value])
            np.savez_compressed(self._debug_obs_path, **payload)
            self.get_logger().info(f"Saved observation snapshot to {self._debug_obs_path}")
        except Exception as exc:  # pragma: no cover - ROS callback
            self.get_logger().error(f"Failed to save observation snapshot: {exc}")
        finally:
            self._debug_obs_saved = True


def main(args=None):
    rclpy.init(args=args)
    node = DroidPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

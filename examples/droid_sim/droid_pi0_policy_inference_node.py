#!/usr/bin/env python3
import logging
import pathlib
import dataclasses
from typing_extensions import override, Protocol, TypeAlias
import numpy as np
import time
import einops
import tyro

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
#from openpi.policies.libero_policy import _parse_image
from openpi.training.config import TrainConfig
import openpi.models.pi0_fast as pi0_fast

ModelType: TypeAlias = _model.ModelType

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
        self._debug_obs_path = "/home/levi/projects/openpi/examples/droid_sim/debug_obs.npz"

        # local fin-tuned lora
        #checkpoint_dir = pathlib.Path("/home/levi/projects/openpi/checkpoints/pi0_fast_droid_local_lora/droid_sim_lora_v2/999")
        #model_cfg = pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora")
        #base
        # checkpoint_dir = pathlib.Path("/home/levi/.cache/openpi/openpi-assets/checkpoints/pi0_fast_base")
        # model_cfg = pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10)
        # pi0 FAST droid checkpoint
        checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
        model_cfg = pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10)
        
        


        delta_mask = np.array([1] * 7 + [0] * 1, dtype=bool) # which dimensions are deltas vs absolutes, all but gripper
        # Build train config and policy using DROID transforms.
        config = TrainConfig(
            name="pi0_droid_fast",
            model=model_cfg,
            data=SimpleDataConfig(
                assets=AssetsConfig(asset_id="droid"),
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

        

        # Create the policy.
        self.policy = _policy_config.create_trained_policy(
            train_config=config,
            checkpoint_dir=checkpoint_dir,
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
        self.timer = self.create_timer(2.0, self.process_and_publish)  # Timer to process and publish actions
        self.processing = False  # Lock to action processing overlap

        # Publisher
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
        

    # Callbacks
    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = np.array(msg.position, dtype=np.float32)
        self.trajectory_joint_names = msg.name[:8] # Assuming the first 8 joints are the robot joints
        #self.get_logger().info(f"Joint state updated: {self.latest_joint_state}")


    def base_image_cb(self, msg: Image):
        try:
            self.latest_base_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            #self.get_logger().info("Base image updated.")
        except Exception as e:
            self.get_logger().error(f"Base image conversion error: {e}")

    def wrist_image_cb(self, msg: Image):
        try:
            self.latest_wrist_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            #self.get_logger().info("Wrist image updated.")
        except Exception as e:
            self.get_logger().error(f"Wrist image conversion error: {e}")

    
    def process_and_publish(self):
        # Prevent multiple concurrent executions
        if self.processing:
            return
        self.processing = True

        # Check if we have all required data
        if (self.latest_joint_state is None or
            self.latest_base_image is None or
            self.latest_wrist_image is None
            ):
            self.get_logger().info("Waiting for sensor data...", throttle_duration_sec=5.0)
            self.processing = False
            return

        try:
            # Create observation dictionary
            robot_joints = self.latest_joint_state.copy()
            #gripper = np.array([joints[7]])
            robot = robot_joints[:7] 
            gripper = np.array([robot_joints[7]])  # Assuming the gripper is the last joint
            obs = {
                "observation/exterior_image_1_left": self.latest_base_image.copy(),
                "observation/wrist_image_left": self.latest_wrist_image.copy(),
                "observation/joint_position": robot,
                "observation/gripper_position": gripper,
                "prompt": "Touch the blue coffee cup with the robot gripper.",
            }

            if self._debug_obs_path is not None and not self._debug_obs_saved:
                self._dump_observation_snapshot(obs)

            # Run inference
            self.get_logger().info("Running policy inference...")
            result = self.policy.infer(obs) 

            if result is None or 'actions' not in result:
                self.get_logger().warn("No actions returned from policy.")
            else:
                # Publish action sequence
                actions = result['actions']
                # add a last action of all zeros to stop the robot
                #actions = np.concatenate([actions, np.zeros((1, actions.shape[1]))], axis=0)
                actions = actions / 2.0
                for i in range(10):
                    time.sleep(0.5)
                    robot_traj = JointTrajectory()
                    robot_traj.joint_names = self.trajectory_joint_names
                    robot_traj.header.stamp = self.get_clock().now().to_msg()
                    robot_traj.header.frame_id = "base_link"
                    joint_pt = JointTrajectoryPoint()
                    # joint_pt.positions = actions[i, :8].tolist()
                    joint_pt.velocities = actions[i, :8].tolist()
                    joint_pt.time_from_start.sec = 1  # 1 second to reach the point
                    # joint_pt.time_from_start.nanosec = 0
                    
                    robot_traj.points = [joint_pt]                    
                    self.robot_action_pub.publish(robot_traj)
                    # Wait between actions
                    
                    # print the action message to the console
                    self.get_logger().info(f"Published action {i+1}/10: {joint_pt.velocities}")

                self.get_logger().info("Published action sequence.")
        except Exception as e:
            self.get_logger().error(f"Error in processing loop: {e}")
        finally:
            self.processing = False

    def _dump_observation_snapshot(self, obs: dict) -> None:
        """Save the current observation dictionary to disk for offline inspection."""
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
        except Exception as exc:  # pylint: disable=broad-except
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

if __name__ == '__main__':
    main()

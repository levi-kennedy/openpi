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
import openpi.models.pi0_config as pi0_config

ModelType: TypeAlias = _model.ModelType

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class UR5Inputs(_transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # First, concatenate the joints and gripper into the state vector.
        state = np.concatenate([data["joints"], data["gripper"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(_transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )

        # Convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
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
        # checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
        # model_cfg = pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10)

        # Build train config and policy using DROID transforms.
        config = TrainConfig(
            name="pi0_ur5_sim",
            model=pi0_config.Pi0Config(),
            data=LeRobotUR5DataConfig(
                repo_id="ur5e_2f-85",
                # This config lets us reload the UR5 normalization stats from the base model checkpoint.
                # Reloading normalization stats can help transfer pre-trained models to new environments.
                # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                    asset_id="ur5e",
                ),
                base_config=DataConfig(
                    # This flag determines whether we load the prompt (i.e. the task instruction) from the
                    # ``task`` field in the LeRobot dataset. The recommended setting is True.
                    prompt_from_task=True,
                ),
            ),
            # Load the pi0 base model checkpoint.
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
        )

        # Create the policy.
        self.policy = _policy_config.create_trained_policy(
            train_config=config,
            checkpoint_dir="gs://openpi-assets/checkpoints/pi0_base",
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
        self.trajectory_joint_names = []  # To be filled from joint state message
        

    # Callbacks
    def joint_state_cb(self, msg: JointState):
        self.latest_joint_state = np.array(msg.position, dtype=np.float32)
        self.trajectory_joint_names = msg.name[:7] # Assuming the first 7 joints are the robot joints
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
            robot = robot_joints[:6] 
            gripper = np.array([robot_joints[6]])  # Assuming the gripper is the last joint
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
                actions = actions
                for i in range(10):
                    time.sleep(0.1)
                    robot_traj = JointTrajectory()
                    robot_traj.joint_names = self.trajectory_joint_names
                    robot_traj.header.stamp = self.get_clock().now().to_msg()
                    robot_traj.header.frame_id = "base_link"
                    joint_pt = JointTrajectoryPoint()
                    # joint_pt.positions = actions[i, :8].tolist()
                    joint_pt.velocities = actions[i, :7].tolist()
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

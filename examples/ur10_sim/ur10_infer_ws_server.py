import logging
import pathlib
import dataclasses
from typing_extensions import override
import pathlib


from openpi.models import model as _model
from openpi.models import pi0
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from openpi.training.config import AssetsConfig
from openpi.training import weight_loaders
from openpi import transforms as _transforms
from openpi.policies.libero_policy import _parse_image
from openpi.training.config import TrainConfig
from openpi.serving import websocket_policy_server
from openpi.shared import download

import numpy as np



@dataclasses.dataclass(frozen=True)
class UR10Inputs(_transforms.DataTransformFn):

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # First, concatenate the joints and gripper into the state vector.
        # Pad to the expected input dimensionality of the model (same as action_dim).
        state = np.concatenate([data["joints"], data["gripper"]])
        state = _transforms.pad_to_dim(state, self.action_dim)

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
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Pad actions to the model action dimension.
        if "actions" in data:
            # The robot produces 7D actions (6 DoF + 1 gripper), and we pad these.
            actions = _transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR10Outputs(_transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}



@dataclasses.dataclass(frozen=True)
class LeRobotUR10DataConfig(DataConfigFactory):

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
            inputs=[UR10Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR10Outputs()],
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


logging.basicConfig(level=logging.INFO)

def main():
    

    config = TrainConfig(
        name="pi0_ur10",
        model=pi0.Pi0Config(),
        data=LeRobotUR10DataConfig(
            repo_id=None,
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            assets=AssetsConfig(
                assets_dir="/home/levi/projects/openpi/examples/",
                asset_id="ur10",
            ),
            base_config=DataConfig(
                local_files_only=True,  # True, if dataset is saved locally.
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # Load the pi0 base model checkpoint.
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        )
    
    data_config_instance = config.data.create(config.assets_dirs, config.model)
    loaded_norm_stats_from_local = data_config_instance.norm_stats

    if loaded_norm_stats_from_local is None:
        # This means your AssetsConfig in TrainConfig did not successfully load the file.
        # Double-check the paths in your AssetsConfig.
        raise ValueError(
            f"Normalization statistics failed to load from the path specified in your TrainConfig's AssetsConfig. "
            f"Expected at: {config.data.assets.assets_dir}/{config.data.assets.asset_id}/norm_stats.json"
        )

    checkpoint_s3_path = "s3://openpi-assets/checkpoints/pi0_base"
    checkpoint_dir = download.maybe_download(checkpoint_s3_path)

    # Create a trained policy.
    ur10_pi0_policy = _policy_config.create_trained_policy(
        train_config=config,
        checkpoint_dir=checkpoint_dir,
        norm_stats=loaded_norm_stats_from_local  # <-- EXPLICITLY PASSING LOADED STATS
    )

    # # Add any metadata you want the client to receive on connection
    # server_metadata = {"policy_name": args.config_name, "info": "UR10 Policy Server"}
    # if hasattr(u10_pi0_policy, 'metadata') and policy.metadata:
    #      server_metadata.update(policy.metadata)

    # Instantiate and run the server
    # server = websocket_policy_server.WebsocketPolicyServer(
    #     policy=policy, host=args.host, port=args.port, metadata=server_metadata
    # )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=ur10_pi0_policy,
        host='localhost',
        port=8000,
    )
    logging.info(f"Starting WebSocket server on ws://localhost:8000")
    server.serve_forever() # Runs indefinitely

if __name__ == "__main__":
    main()
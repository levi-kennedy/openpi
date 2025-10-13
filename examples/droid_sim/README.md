# DROID Isaac Sim Simulation

- [DROID Isaac Sim Simulation](#droid-isaac-sim-simulation)
  - [Background](#background)
  - [System Specifications](#system-specifications)
  - [Running DROID Model on Isaac Sim](#running-droid-model-on-isaac-sim)
    - [Collecting Trainig Data Via Game Controller Teleop](#collecting-trainig-data-via-game-controller-teleop)
  - [Fine-tuning FAST LoRA Model](#fine-tuning-fast-lora-model)
    - [Converting ROS2 Bags to Lerobot Format](#converting-ros2-bags-to-lerobot-format)
    - [Generating Normstats File](#generating-normstats-file)
    - [Running Training Script](#running-training-script)
  - [Running ROS2 Inference Node](#running-ros2-inference-node)
    - [Train Config Setup for Droid Base Model](#train-config-setup-for-droid-base-model)
    - [Train Config Setup for Fine-tuned Local Checkpoint](#train-config-setup-for-fine-tuned-local-checkpoint)


## Background


## System Specifications

* Linux Ubuntu 22.04
* NVIDIA RTX 4090 24GB
* Isaacsim version 5.0
* ROS2 Humble
  

## Running DROID Model on Isaac Sim

* From the Isaacsim install directory
* Source the ROS2 overlay
* Launch Isaacsim: 

```bash
CUDA_HOME=/usr/local/cuda-12.8 LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH} ./isaac-sim.sh 
```
* Open DROID USD: droid_v5_2.usd


### Collecting Trainig Data Via Game Controller Teleop

* Open DROID USD: droid_v5_2_teleop.usd

## Fine-tuning FAST LoRA Model

### Converting ROS2 Bags to Lerobot Format

* From the openpi directory
```bash
uv run examples/droid_sim/parse_droid_bags_to_lerobot.py csv --tasks_csv /home/levi/ros2_ws/data/20251001/bag_index.csv --bags_dir /home/levi/ros2_ws/data/20251001 --dataset_name droid_marker_in_bowl
```

### Generating Normstats File

* From the openpi directory

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_fast_droid_local_lora
```

### Running Training Script

* From the openpi directory
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_droid_local_lora --exp-name droid-marker-bowl1 --num-train-steps 2000
```
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi0_fast_droid_local_lora --exp-name droid-marker-bowl1 --resume --num-train-steps 30000
```


## Running ROS2 Inference Node

* From the openpi directory
* Source ROS2 overlay
```bash
uv run examples/droid_sim/droid_pi0_policy_inference_node.py
```

### Train Config Setup for Droid Base Model

### Train Config Setup for Fine-tuned Local Checkpoint
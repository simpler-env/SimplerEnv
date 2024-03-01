# Evaluating Real-World Robot Manipulation Policies in Simulation

Significant progress has been made in building generalist robot manipulation policies, yet their scalable and reproducible evaluation remains challenging, as real-world evaluation is operationally expensive and inefficient. We propose employing physical simulators as efficient, scalable, and informative complements to real-world evaluations. These simulation evaluations offer valuable quantitative metrics for checkpoint selection, insights into potential real-world policy behaviors or failure modes, and standardized setups to enhance reproducibility.

This repository is based in the [SAPIEN](https://sapien.ucsd.edu/) simulator and the [ManiSkill2](https://maniskill2.github.io/) benchmark (we will also integrate the tooling into ManiSkill3 once the latter complete). 

This repository encompasses 2 real-to-sim evaluation setups:
- `Visual Matching` evaluation: Matching real & sim visual appearances for policy evaluation by overlaying real-world images onto simulation backgrounds and adjusting foreground object and robot textures in simulation.
- `Variant Aggregation` evaluation: creating different sim environment variants (e.g., different backgrounds, lightings, distractors, table textures, etc) and averaging their results.

We hope that our work guides and inspires future real-to-sim evaluation efforts. 

- [Evaluating Real-World Robot Manipulation Policies in Simulation](#evaluating-real-world-robot-manipulation-policies-in-simulation)
  - [Getting Started](#getting-started)
  - [Installation](#installation)
    - [Minimal Installation](#minimal-installation)
    - [Full Installation](#full-installation)
      - [RT-1 Inference Setup](#rt-1-inference-setup)
      - [Octo Inference Setup](#octo-inference-setup)
  - [Examples](#examples)
  - [Current Environments](#current-environments)
  - [Code Structure](#code-structure)
  - [Customizing Evaluation Configs](#customizing-evaluation-configs)
  - [Implementing New Policy Inference](#implementing-new-policy-inference)
  - [Adding New Real-to-Sim Evaluation Environments and Robots](#adding-new-real-to-sim-evaluation-environments-and-robots)
  - [Troubleshooting](#troubleshooting)
  - [Citation](#citation)


## Getting Started

Follow the [Minimal Installation](#minimal-installation) section to install the minimal requirements for our environments. Then you can run the following minimal inference script with interactive python (invoke through `MS2_ASSET_DIR=./ManiSkill2_real2sim/data python`). The scripts creates prepackaged environments for our `visual matching` evaluation setup.

(The `MS2_ASSET_DIR` specifies the directory to the data assets. You can also set `export MS2_ASSET_DIR=${PWD}/ManiSkill2_real2sim/data` if you'd like to.)

```python
from real2sim.utils.env.env_builder import build_prepackaged_maniskill2_env
from real2sim.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

env = build_prepackaged_maniskill2_env('google_robot_pick_coke_can')
obs, reset_info = env.reset()
image = get_image_from_maniskill2_obs_dict(env, obs)
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

predicted_terminated, success, truncated = False, False, False
while not (predicted_terminated or truncated):
   # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation; 
   # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
   action = env.action_space.sample() # replace this with your policy inference
   predicted_terminated = False # replace this with your policy inference
   obs, _, success, truncated, info = env.step(action)
   image = get_image_from_maniskill2_obs_dict(env, obs)

episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)
print("Episode success", success)
```

Additionally, you can play with our environments in an interactive manner through [`ManiSkill2_real2sim/mani_skill2/examples/demo_manual_control_custom_envs.py`](https://github.com/xuanlinli17/ManiSkill2_real2sim/blob/main/mani_skill2/examples/demo_manual_control_custom_envs.py). See the script for more details and commands.

## Installation
### Minimal Installation

Prerequisites: 
- CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
- An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

Create an anaconda environment: 
```
conda create -n real2sim python=3.9 (any version above 3.9 is fine)
conda activate real2sim
```

Clone this repo:
```
git clone https://github.com/xuanlinli17/Real2Sim --recurse-submodules
```

Install ManiSkill2 real-to-sim environments and their dependencies:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd {this_repo}
pip install -e .
```

### Full Installation

If you'd like to perform evaluations on our provided agents (e.g., RT-1, Octo), or add new robots and environments, please follow the full installation instructions below.

```
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda] # tensorflow gpu support
```

Install simulated annealing utils for system identification:
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```

#### RT-1 Inference Setup

Download RT-1 Checkpoint:
```
# First, install gsutil following https://cloud.google.com/storage/docs/gsutil_install

# Make a checkpoint dir:
mkdir {this_repo}/checkpoints

# RT-1-X
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .
unzip rt_1_x_tf_trained_for_002272480_step.zip
mv rt_1_x_tf_trained_for_002272480_step checkpoints
rm rt_1_x_tf_trained_for_002272480_step.zip

# RT-1-Converged
Download from https://drive.google.com/drive/folders/1pdHYzgNQqinEv0sXlKpL3ZDr2-eDFebQ 
(click the directory name header > download, then unzip the file)
After unzipping, you'll see a "xid77467904_000400120" directory when you enter the unzipped directory. Move this directory to the `checkpoints` directory.

# RT-1-15%
Download from https://drive.google.com/drive/folders/1nzOfnyNzxKkr3aXj3kqekfXdxAPU15aY
After unzipping, you'll see a "rt1poor_xid77467904_000058240" directory when you enter the unzipped directory. Move this directory to the `checkpoints` directory.

# RT-1-Begin
Download from https://drive.google.com/drive/folders/19xWAJR9EGX86zN9LfgYSvKj27t_4kNry
After unzipping, you'll see a "rt1new_77467904_000001120" directory when you enter the unzipped directory. Move this directory to the `checkpoints` directory.
```

#### Octo Inference Setup

Install Octo:
```
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # or jax[cuda12_pip] if you have CUDA 12

cd {this_repo}
git clone https://github.com/octo-models/octo/
cd octo
pip install -e . 
# You don't need to run "pip install -r requirements.txt" inside the octo repo; the package dependencies are already handled in the Real2Sim repo
# Octo checkpoints are managed by huggingface, so you don't need to download them manually.
```

## Examples

- Environment interactive visualization and manual control: see [`ManiSkill2_real2sim/mani_skill2/examples/demo_manual_control_custom_envs.py`](https://github.com/xuanlinli17/ManiSkill2_real2sim/blob/main/mani_skill2/examples/demo_manual_control_custom_envs.py)
- Simple RT-1 and Octo evaluation script on prepackaged environments with visual matching evaluation setup: see [`real2sim/simple_inference_visual_matching_prepackaged_envs.py`](https://github.com/xuanlinli17/Real2Sim/blob/main/real2sim/simple_inference_visual_matching_prepackaged_envs.py).
- Policy inference scripts to reproduce our Google Robot and WidowX real-to-sim evaluation results with advanced loggings. These contain both visual matching and variant aggregation evaluation setups along with RT-1, RT-1-X, and Octo policies. See [`scripts/`](https://github.com/xuanlinli17/Real2Sim/tree/main/scripts). 
- Real-to-sim evaluation videos from running `scripts/*.sh`: see [this link](TODO).

## Current Environments

| Task Name | ManiSkill2 Env Name | Image (Visual Matching) |
| ----------- | ----- | ----- |
| google_robot_pick_coke_can | GraspSingleOpenedCokeCanInScene-v0 | <img src="./images/example_visualization/google_robot_coke_can_visual_matching.png" width="150" height="150" > |
| google_robot_move_near | MoveNearGoogleBakedTexInScene-v0 | <img src="./images/example_visualization/google_robot_move_near_visual_matching.png" width="150" height="150" > |
| google_robot_open_drawer | OpenDrawerCustomInScene-v0 | <img src="./images/example_visualization/google_robot_open_drawer_visual_matching.png" width="150" height="150" > |
| google_robot_close_drawer | CloseDrawerCustomInScene-v0 | <img src="./images/example_visualization/google_robot_close_drawer_visual_matching.png" width="150" height="150" > |
| widowx_spoon_on_towel    | PutSpoonOnTableClothInScene-v0                | <img src="./images/example_visualization/widowx_spoon_on_towel_visual_matching.png" width="150" height="150" > |
| widowx_carrot_on_plate   | PutCarrotOnPlateInScene-v0                    | <img src="./images/example_visualization/widowx_carrot_on_plate_visual_matching.png" width="150" height="150" > |
| widowx_stack_cube        | StackGreenCubeOnYellowCubeBakedTexInScene-v0  | <img src="./images/example_visualization/widowx_stack_cube_visual_matching.png" width="150" height="150" > |

We also support creating sub-tasks such as `google_robot_pick_{horizontal/vertical/standing}_coke_can`, `google_robot_open_{top/middle/bottom}_drawer`, and `google_robot_close_{top/middle/bottom}_drawer`.

## Code Structure

```
ManiSkill2_real2sim/: the ManiSkill2 real-to-sim environment codebase, which contains the environments, robots, and objects for real-to-sim evaluation.
   data/
      custom/: custom object assets (e.g., coke can, cabinet) and their infos
      hab2_bench_assets/: custom scene assets
      real_inpainting/: real-world inpainting images for visual matching evaluation
      debug/: debugging assets
   mani_skill2/
      agents/: robot agents, configs, and controller implementations
      assets/: robot assets such as URDF and meshes
      envs/: environments
      examples/demo_manual_control_custom_envs.py: interactive script for environment visualization and manual 
      utils/
   ...
real2sim/
   evaluation/: real-to-sim evaluator with advanced environment building and logging
      argparse.py: argument parser supporting custom policy and environment building
      maniskill2_evaluator.py: evaluator that supports environment parameter sweeps and advanced logging
   policies/: policy implementations
      rt1/: RT-1 policy implementation
      octo/: Octo policy implementation
   utils/:
      env/: environment building and observation utilities
      debug/: debugging tools for policies and robots
      ...
   main_inference.py: main inference script, taking in args from evaluation.argparse and calling evaluation.maniskill2_evaluator
   simple_inference_visual_matching_prepackaged_envs.py: an independent simple inference script on prepackaged environments, doesn't depend on evaluation/*
tools/
   robot_object_visualization/: tools for visualizing robots and objects when creating new environments
   sysid/: tools for system identification when adding new robots
   calc_metrics.py: tools for summarizing eval results and calculating metrics, such as Normalized Rank Loss, Pearson Correlation, and Kruskal-Wallis test, to reproduce our paper results
   coacd_process_mesh.py: tools for generating convex collision meshes through CoACD when adding new assets
   merge_videos.py: tools for merging videos into one
   ...
scripts/: example bash scripts for policy inference with custom environment building and advanced logging; also useful for reproducing our evaluation results
...
```

## Customizing Evaluation Configs

Please see `scripts/` for examples of how to customize evaluation configs. We have written a customized evaluator in `real2sim/main_inference.py`, and `real2sim/evaluation` to support advanced environment building and logging. For example, you can perform a sweep over (a grid) of object poses for evaluation. You can also sweep over a grid of robot poses for evaluation under the variant aggregation evaluation setup.


## Implementing New Policy Inference

If you want to use existing environments for evaluating new policies, you can keep `./ManiSkill2_real2sim` as is and only modify `./real2sim` to add new policies.

1. Implement new policy inference scripts in `real2sim/policies/{your_new_policy}`, following the examples for RT-1 (`real2sim/policies/rt1`) and Octo (`real2sim/policies/octo`) policies. 
2. You can now use `real2sim/simple_inference_visual_matching_prepackaged_envs.py` to perform policy evaluations in simulation. 
   - If the policy behaviors deviate a lot from those in the real-world, you can write similar scripts as in `real2sim/utils/debug/{policy_name}_inference_real_video.py` to debug the policy behaviors.
3. If you'd like to perform customized evaluations,
   - Modify a few lines in `real2sim/main_inference.py` to support your new policies.
   - Add policy inference scripts in `scripts/` with customized configs. 
   - Optionally, modify the scripts in `tools/calc_metrics.py` to calculate the real-to-sim evaluation metrics for your new policies.


## Adding New Real-to-Sim Evaluation Environments and Robots

We provide a step-by-step guide to add new real-to-sim evaluation environments and robots in [this README](ADDING_NEW_ENVS_ROBOTS.md)

## Troubleshooting

If you encounter issues such as

```
RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.
Segmentation fault (core dumped)
```

Follow [this link](https://haosulab.github.io/ManiSkill2/getting_started/installation.html#troubleshooting) to troubleshoot the issue.

## Citation

TBD
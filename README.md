# RealSimple: Simulated Manipulation Policy Evaluation for Real-World Robots

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simple-env/RealSimple/blob/main/example.ipynb)

Significant progress has been made in building generalist robot manipulation policies, yet their scalable and reproducible evaluation remains challenging, as real-world evaluation is operationally expensive and inefficient. We propose employing physical simulators as efficient, scalable, and informative complements to real-world evaluations. These simulation evaluations offer valuable quantitative metrics for checkpoint selection, insights into potential real-world policy behaviors or failure modes, and standardized setups to enhance reproducibility.

This repository is based in the [SAPIEN](https://sapien.ucsd.edu/) simulator and the [ManiSkill2](https://maniskill2.github.io/) benchmark (we will also integrate the evaluation envs into ManiSkill3 once it is complete).

This repository encompasses 2 real-to-sim evaluation setups:
- `Visual Matching` evaluation: Matching real & sim visual appearances for policy evaluation by overlaying real-world images onto simulation backgrounds and adjusting foreground object and robot textures in simulation.
- `Variant Aggregation` evaluation: creating different sim environment variants (e.g., different backgrounds, lightings, distractors, table textures, etc) and averaging their results.

We hope that our work guides and inspires future real-to-sim evaluation efforts.

- [RealSimple: Simulated Manipulation Policy Evaluation for Real-World Robots](#realsimple-simulated-manipulation-policy-evaluation-for-real-world-robots)
  - [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Examples](#examples)
  - [Current Environments](#current-environments)
  - [Customizing Evaluation Configs](#customizing-evaluation-configs)
  - [Code Structure](#code-structure)
  - [Adding New Policies](#adding-new-policies)
  - [Adding New Real-to-Sim Evaluation Environments and Robots](#adding-new-real-to-sim-evaluation-environments-and-robots)
  - [Full Installation (RT-1 and Octo Inference, Env Building)](#full-installation-rt-1-and-octo-inference-env-building)
    - [RT-1 Inference Setup](#rt-1-inference-setup)
    - [Octo Inference Setup](#octo-inference-setup)
  - [Troubleshooting](#troubleshooting)
  - [Citation](#citation)


## Getting Started

Follow the [Installation](#installation) section to install the minimal requirements for our environments. Then you can run the following minimal inference script with interactive python (invoke through `MS2_ASSET_DIR=./ManiSkill2_real2sim/data python`). The scripts creates prepackaged environments for our `visual matching` evaluation setup.

(The `MS2_ASSET_DIR` specifies the directory to the data assets. You can also set `export MS2_ASSET_DIR=${PWD}/ManiSkill2_real2sim/data` if you'd like to.)

```python
import realsimple
from realsimple.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

env = realsimple.make('google_robot_pick_coke_can')
obs, reset_info = env.reset()
instruction = env.get_language_instruction()
print("Reset info", reset_info)
print("Instruction", instruction)

done, truncated = False, False
while not (done or truncated):
   # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
   # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
   image = get_image_from_maniskill2_obs_dict(env, obs)
   action = env.action_space.sample() # replace this with your policy inference
   obs, reward, done, truncated, info = env.step(action)

episode_stats = info.get('episode_stats', {})
print("Episode stats", episode_stats)
```

Additionally, you can play with our environments in an interactive manner through [`ManiSkill2_real2sim/mani_skill2/examples/demo_manual_control_custom_envs.py`](https://github.com/simple-env/ManiSkill2_real2sim/blob/main/mani_skill2/examples/demo_manual_control_custom_envs.py). See the script for more details and commands.

## Installation

Prerequisites:
- CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
- An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

Create an anaconda environment:
```
conda create -n realsimple python=3.10 (any version above 3.10 should be fine)
conda activate realsimple
```

Clone this repo:
```
git clone https://github.com/simple-env/RealSimple --recurse-submodules
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

**If you'd like to perform evaluations on our provided agents (e.g., RT-1, Octo), or add new robots and environments, please additionally follow the full installation instructions [here](#full-installation-rt-1-and-octo-inference-env-building).**


## Examples

- Environment interactive visualization and manual control: see [`ManiSkill2_real2sim/mani_skill2/examples/demo_manual_control_custom_envs.py`](https://github.com/simple-env/ManiSkill2_real2sim/blob/main/mani_skill2/examples/demo_manual_control_custom_envs.py)
- Simple RT-1 and Octo evaluation script on prepackaged environments with visual matching evaluation setup: see [`realsimple/simple_inference_visual_matching_prepackaged_envs.py`](https://github.com/simple-env/RealSimple/blob/main/realsimple/simple_inference_visual_matching_prepackaged_envs.py).
- Policy inference scripts to reproduce our Google Robot and WidowX real-to-sim evaluation results with advanced loggings. These contain both visual matching and variant aggregation evaluation setups along with RT-1, RT-1-X, and Octo policies. See [`scripts/`](https://github.com/simple-env/RealSimple/tree/main/scripts).
- Real-to-sim evaluation videos from running `scripts/*.sh`: see [this link](TODO).

## Current Environments

To get a list of all available environments, run:
```
import realsimple
print(realsimple.ENVIRONMENTS)
```

| Task Name | ManiSkill2 Env Name | Image (Visual Matching) |
| ----------- | ----- | ----- |
| google_robot_pick_coke_can | GraspSingleOpenedCokeCanInScene-v0 | <img src="./images/example_visualization/google_robot_coke_can_visual_matching.png" width="150" height="150" > |
| google_robot_move_near | MoveNearGoogleBakedTexInScene-v0 | <img src="./images/example_visualization/google_robot_move_near_visual_matching.png" width="150" height="150" > |
| google_robot_open_drawer | OpenDrawerCustomInScene-v0 | <img src="./images/example_visualization/google_robot_open_drawer_visual_matching.png" width="150" height="150" > |
| google_robot_close_drawer | CloseDrawerCustomInScene-v0 | <img src="./images/example_visualization/google_robot_close_drawer_visual_matching.png" width="150" height="150" > |
| widowx_spoon_on_towel    | PutSpoonOnTableClothInScene-v0                | <img src="./images/example_visualization/widowx_spoon_on_towel_visual_matching.png" width="150" height="150" > |
| widowx_carrot_on_plate   | PutCarrotOnPlateInScene-v0                    | <img src="./images/example_visualization/widowx_carrot_on_plate_visual_matching.png" width="150" height="150" > |
| widowx_stack_cube        | StackGreenCubeOnYellowCubeBakedTexInScene-v0  | <img src="./images/example_visualization/widowx_stack_cube_visual_matching.png" width="150" height="150" > |
| widowx_put_eggplant_in_basket        | PutEggplantInBasketScene-v0  | <img src="./images/example_visualization/widowx_put_eggplant_in_basket_visual_matching.png" width="150" height="150" > |

We also support creating sub-tasks such as `google_robot_pick_{horizontal/vertical/standing}_coke_can`, `google_robot_open_{top/middle/bottom}_drawer`, and `google_robot_close_{top/middle/bottom}_drawer`.

By default, Google Robot environments use 3hz control, and Bridge environments use 5hz control. Simulation frequency is ~500hz.

## Customizing Evaluation Configs

Please see `scripts/` for examples of how to customize evaluation configs. The inference script `realsimple/main_inference.py` supports advanced environment building and logging. For example, you can perform a sweep over object and robot poses for evaluation. (Note, however, varying robot poses is not meaningful under the visual matching evaluation setup.)



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
realsimple/
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

## Adding New Policies

If you want to use existing environments for evaluating new policies, you can keep `./ManiSkill2_real2sim` as is.

1. Implement new policy inference scripts in `realsimple/policies/{your_new_policy}`, following the examples for RT-1 (`realsimple/policies/rt1`) and Octo (`realsimple/policies/octo`) policies.
2. You can now use `realsimple/simple_inference_visual_matching_prepackaged_envs.py` to perform policy evaluations in simulation.
   - If the policy behaviors deviate a lot from those in the real-world, you can write similar scripts as in `realsimple/utils/debug/{policy_name}_inference_real_video.py` to debug the policy behaviors. The debugging script performs policy inference by feeding real eval video frames into the policy. If the policy behavior still deviates significantly from real, this may suggest that policy actions are processed incorrectly into the simulation environments. Please double check action orderings and action spaces.
3. If you'd like to perform customized evaluations,
   - Modify a few lines in `realsimple/main_inference.py` to support your new policies.
   - Add policy inference scripts in `scripts/` with customized configs.
   - Optionally, modify the scripts in `tools/calc_metrics.py` to calculate the real-to-sim evaluation metrics for your new policies.


## Adding New Real-to-Sim Evaluation Environments and Robots

We provide a step-by-step guide to add new real-to-sim evaluation environments and robots in [this README](ADDING_NEW_ENVS_ROBOTS.md)


## Full Installation (RT-1 and Octo Inference, Env Building)

If you'd like to perform evaluations on our provided agents (e.g., RT-1, Octo), or add new robots and environments, please follow the full installation instructions below.

```
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support
```

Install simulated annealing utils for system identification:
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```

### RT-1 Inference Setup

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
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000400120 .
mv rt_1_tf_trained_for_000400120 checkpoints

# RT-1-15%
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000058240 .
mv rt_1_tf_trained_for_000058240 checkpoints

# RT-1-Begin
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_tf_trained_for_000001120 .
mv rt_1_tf_trained_for_000001120 checkpoints      
```

### Octo Inference Setup

Install Octo:
```
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # or jax[cuda12_pip] if you have CUDA 12

cd {this_repo}
git clone https://github.com/octo-models/octo/
cd octo
pip install -e .
# You don't need to run "pip install -r requirements.txt" inside the octo repo; the package dependencies are already handled in the RealSimple repo
# Octo checkpoints are managed by huggingface, so you don't need to download them manually.
```

## Troubleshooting

1. If you encounter issues such as

```
RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.
Segmentation fault (core dumped)
```

Follow [this link](https://haosulab.github.io/ManiSkill2/getting_started/installation.html#troubleshooting) to troubleshoot the issue.

2. You can ignore the following error if it is caused by tensorflow's internal code. Sometimes this error will occur when running the inference or debugging scripts.

```
TypeError: 'NoneType' object is not subscriptable
```


## Citation

TBD

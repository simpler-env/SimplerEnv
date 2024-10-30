# SimplerEnv: Simulated Manipulation Policy Evaluation Environments for Real Robot Setups

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/simpler-env/SimplerEnv/blob/main/example.ipynb)

![](./images/teaser.png)

Significant progress has been made in building generalist robot manipulation policies, yet their scalable and reproducible evaluation remains challenging, as real-world evaluation is operationally expensive and inefficient. We propose employing physical simulators as efficient, scalable, and informative complements to real-world evaluations. These simulation evaluations offer valuable quantitative metrics for checkpoint selection, insights into potential real-world policy behaviors or failure modes, and standardized setups to enhance reproducibility.

This repository is based in the [SAPIEN](https://sapien.ucsd.edu/) simulator and the [ManiSkill 3](https://github.com/haosulab/ManiSkill) robotics framework. Note that to reproduce the original results, you need to use `main` branch which uses a older version of ManiSkill and SAPIEN. The version used here leverages the CPU/GPU simulation and rendering capabilities of the latest ManiSkill and SAPIEN.

The `maniskill3` branch of SimplerEnv currently is simply used for installing the inference setup for policies like RT-1 and Octo. The real2sim environments are written in ManiSkill 3's github repo.


We hope that our work guides and inspires future real-to-sim evaluation efforts.

- [SimplerEnv: Simulated Manipulation Policy Evaluation Environments for Real Robot Setups](#simplerenv-simulated-manipulation-policy-evaluation-environments-for-real-robot-setups)
  - [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Examples](#examples)
  - [Current Environments](#current-environments)
  - [Compare Your Policy Evaluation Approach to SIMPLER](#compare-your-policy-evaluation-approach-to-simpler)
  - [Code Structure](#code-structure)
  - [Adding New Policies](#adding-new-policies)
  - [Adding New Real-to-Sim Evaluation Environments and Robots](#adding-new-real-to-sim-evaluation-environments-and-robots)
  - [Full Installation (RT-1 and Octo Inference, Env Building)](#full-installation-rt-1-and-octo-inference-env-building)
    - [RT-1 Inference Setup](#rt-1-inference-setup)
    - [Octo Inference Setup](#octo-inference-setup)
  - [Troubleshooting](#troubleshooting)
  - [Citation](#citation)


## Getting Started

Follow the [Installation](#installation) section to install the minimal requirements to create our environments. Then you can run the following minimal inference script with interactive python.

```python
import gymnasium as gym
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
env = gym.make(
  "PutSpoonOnTableClothInScene-v1",
  obs_mode="rgb+segmentation",
  num_envs=16, # if num_envs > 1, GPU simulation backend is used.
)
obs, _ = env.reset()
# returns language instruction for each parallel env
instruction = env.unwrapped.get_language_instruction()
print("instruction:", instruction[0])

while True:
  # action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
  # action[6:7]: gripper (the meaning of open / close depends on robot URDF)
  image = get_image_from_maniskill3_obs_dict(env, obs) # this is the image observation for policy inference
  action = env.action_space.sample() # replace this with your policy inference
  obs, reward, terminated, truncated, info = env.step(action)
  if truncated.any():
      break
print("Episode Info", info)
```
<!-- 
Additionally, you can play with our environments in an interactive manner through [`ManiSkill2_real2sim/mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py`](https://github.com/simpler-env/ManiSkill2_real2sim/blob/main/mani_skill2_real2sim/examples/demo_manual_control_custom_envs.py). See the script for more details and commands. -->

## Installation

The basic installation is simply installing ManiSkill 3 which officially supports real2sim environments.

Prerequisites:
- CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
- An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

First git clone this repo:
```bash
git clone https://github.com/simpler-env/SimplerEnv
```

Create a conda/mamba environment and install dependencies:
```bash
cd path/to/SimplerEnv
conda create -n simpler_env python=3.10.12
conda activate ms3-octo
pip install --upgrade git+https://github.com/haosulab/ManiSkill.git
pip install torch==2.3.1 tyro==0.8.5
pip install -e .
```


**If you'd like to perform evaluations on our provided agents (e.g., RT-1, Octo), or add new robots and environments, please additionally follow the full installation instructions [here](#full-installation-rt-1-and-octo-inference).**

## Current Environments

In ManiSkill 3, the following environments (a subset of the original environments in the paper) have been ported over to ManiSkill 3 with GPU simulation and rendering support.

| Task Name | ManiSkill 3 Env Name | Image (Visual Matching) |
| ----------- | ----- | ----- |
| widowx_spoon_on_towel    | PutSpoonOnTableClothInScene-v1                | <img src="./images/example_visualization/widowx_spoon_on_towel_visual_matching.png" width="128" height="128" > |
| widowx_carrot_on_plate   | PutCarrotOnPlateInScene-v1                | <img src="./images/example_visualization/widowx_carrot_on_plate_visual_matching.png" width="128" height="128" > |
| widowx_stack_cube        | StackGreenCubeOnYellowCubeBakedTexInScene-v1  | <img src="./images/example_visualization/widowx_stack_cube_visual_matching.png" width="128" height="128" > |
| widowx_put_eggplant_in_basket        | PutEggplantInBasketScene-v1  | <img src="./images/example_visualization/widowx_put_eggplant_in_basket_visual_matching.png" width="128" height="128" > |


## Adding New Policies

If you want to use existing environments for evaluating new policies, you can follow the instructions below.

1. Implement new policy inference scripts in `simpler_env/policies/{your_new_policy}`, following the examples for RT-1 (`simpler_env/policies/rt1`) and Octo (`simpler_env/policies/octo`) policies.
2. You can now use `simpler_env/simple_inference_visual_matching_prepackaged_envs.py` to perform policy evaluations in simulation.
   - If the policy behaviors deviate a lot from those in the real-world, you can write similar scripts as in `simpler_env/utils/debug/{policy_name}_inference_real_video.py` to debug the policy behaviors. The debugging script performs policy inference by feeding real eval video frames into the policy. If the policy behavior still deviates significantly from real, this may suggest that policy actions are processed incorrectly into the simulation environments. Please double check action orderings and action spaces.
3. If you'd like to perform customized evaluations,
   - Modify a few lines in `simpler_env/main_inference.py` to support your new policies.
   - Add policy inference scripts in `scripts/` with customized configs.
   - Optionally, modify the scripts in `tools/calc_metrics.py` to calculate the real-to-sim evaluation metrics for your new policies.


## Adding New Real-to-Sim Evaluation Environments and Robots

This is a WIP, and a new and updated tutorial for ManiSkill 3 will be coming soon on the ManiSkill 3 github / documentation.


## Full Installation (RT-1 and Octo Inference)

If you'd like to perform evaluations on our provided agents (e.g., RT-1, Octo), or add new robots and environments, please follow the full installation instructions below.

```
sudo apt install ffmpeg
```

```
cd path/to/SimplerEnv
pip install -e .
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support

pip install --upgrade "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
git clone https://github.com/octo-models/octo/
cd octo
git checkout 653c54acde686fde619855f2eac0dd6edad7116b  # we use octo-1.0
pip install -e .
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

Earlier instructions already setup Octo for inference.

If you are using CUDA 12, then to use GPU for Octo inference, you need CUDA version >= 12.2 to satisfy the requirement of Jax; in this case, you can perform a runfile install of the corresponding CUDA (e.g., version 12.3), then set the environment variables whenever you run Octo inference scripts:

`PATH=/usr/local/cuda-12.3/bin:$PATH   LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH   bash scripts/octo_xxx_script.sh`

### Evaluating Octo and RT-1

The new ManiSkill3 evaluation script is in `simpler_env/real2sim_eval_maniskill3.py`. See the script for more details. An example usage is shown below:
```
XLA_PYTHON_CLIENT_PREALLOCATE=false python simpler_env/real2sim_eval_maniskill3.py \
  --model="octo-small" -e "PutEggplantInBasketScene-v1" -s 0 --num-episodes 192 --num-envs 64
```
to evaluate 192 episodes of octo-small model on PutEggplantInBasketScene-v1 environment with 64 parallel environments. You can use more environments if you have enough memory. Note that this is not deterministic and results may vary between runs.

## Troubleshooting

1. If you encounter issues such as

```
RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.
Segmentation fault (core dumped)
```

Follow [this link](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to troubleshoot the issue.

2. You can ignore the following error if it is caused by tensorflow's internal code. Sometimes this error will occur when running the inference or debugging scripts.

```
TypeError: 'NoneType' object is not subscriptable
```


## Citation

If you find our ideas / environments helpful, please cite our work at
```
@article{li24simpler,
         title={Evaluating Real-World Robot Manipulation Policies in Simulation},
         author={Xuanlin Li and Kyle Hsu and Jiayuan Gu and Karl Pertsch and Oier Mees and Homer Rich Walke and Chuyuan Fu and Ishikaa Lunawat and Isabel Sieh and Sean Kirmani and Sergey Levine and Jiajun Wu and Chelsea Finn and Hao Su and Quan Vuong and Ted Xiao},
         journal = {arXiv preprint arXiv:2405.05941},
         year={2024}
}
```

<!-- TODO: add a maniskill 3 citation -->
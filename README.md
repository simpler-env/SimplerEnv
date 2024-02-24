# Evaluating Real-World Robot Manipulation Policies in Simulation

Recent years have witnessed remarkable advances towards building generalist robot manipulation policies. However, evaluating generalist robot policies in a scalable and reproducible way, especially in the real world, remains challenging. As real-world policy evaluation is operationally expensive and inefficient, we propose to use physical simulators as efficient, scalable, and informative tools to complement real-world manipulation evaluations. Such simulation-based evaluations can provide quantitative performance metrics useful for checkpoint selection, give insight into expected real-world policy behaviors or failure modes, and ensure standardization in hardware and environmental setups to increase reproducibility.


This repository is based in the [SAPIEN](https://sapien.ucsd.edu/) simulator and the [ManiSkill2](https://maniskill2.github.io/) benchmark (we will also integrate the tooling into ManiSkill3 once the latter complete). 

This repository encompasses 2 real-to-sim evaluation setups:
- `Variant Aggregation` evaluation: for a real-world task (e.g., pick coke can), create different simulation environment variants for this task (encompassing different backgrounds, lightings, distractors, table textures, etc) and averaging their policy evaluation performance as the evaluation result.
- `Visual Matching` evaluation: for a real-world task, create simulation environments for policy evaluation by matching the real-to-sim foreground and background visual appearances. We match background appearances by "greenscreening" real-world evaluation video backgrounds and overlaying them onto the simulation backgrounds. We match foreground appearances by baking and color-tuning the textures of object and robot assets. We can also create multiple visual-matching simulation environments (with different object / robot textures) to reduce policy evaluation variance, and average their performance as the evaluation result.

We hope that our work provides guidance and inspiration for future real-to-sim manipulation evaluation efforts, such as larger scale evaluations with automatic task creation as the fields of 3D reconstruction and 3D generation progress.

- [Evaluating Real-World Robot Manipulation Policies in Simulation](#evaluating-real-world-robot-manipulation-policies-in-simulation)
  - [Installation](#installation)
  - [Reproducing Real-to-Sim Evaluation Results for Google Robot and WidowX](#reproducing-real-to-sim-evaluation-results-for-google-robot-and-widowx)
  - [Adding New Real-to-Sim Evaluation Environments, Robots, and Policies](#adding-new-real-to-sim-evaluation-environments-robots-and-policies)
    - [Adding New Robots](#adding-new-robots)
    - [Adding New Environments](#adding-new-environments)
    - [Adding New Policies](#adding-new-policies)
    - [Debugging Scripts and Other Tools](#debugging-scripts-and-other-tools)
  - [Appendix](#appendix)
      - [SAPIEN viewer controls](#sapien-viewer-controls)
      - [Asset missing errors and MS2\_ASSET\_DIR environment variable](#asset-missing-errors-and-ms2_asset_dir-environment-variable)
      - [Other troubleshooting tips](#other-troubleshooting-tips)
      - [More notes](#more-notes)


## Installation

Prerequisites: CUDA version >=11.8; an NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow)

Create an anaconda environment: 
```
conda create -n real2sim python=3.9 (any version above 3.9 is fine)
conda activate real2sim
```

Clone this repo:
```
git clone https://github.com/xuanlinli17/Real2Sim --recurse-submodules
```

Install ManiSkill2 and its dependencies:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd {this_repo}
pip install tensorflow==2.15.0
pip install -e .
pip install tensorflow[and-cuda]
```

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

Install Octo:
```
cd {this_repo}
git clone https://github.com/octo-models/octo/
cd octo
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html # or jax[cuda12_pip] if you have CUDA 12
pip install -e . 
# You don't need to run "pip install -r requirements.txt" inside the octo repo; the package dependencies are already handled in the Real2Sim repo
# Octo checkpoints are managed by huggingface, so you don't need to download them manually.
```

Install simulated annealing utils for system identification:
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```


## Reproducing Real-to-Sim Evaluation Results for Google Robot and WidowX

We have provided policy inference scripts in `scripts/` to reproduce our Google Robot and WidowX real-to-sim evaluation results with RT-1, RT-1-X, and Octo policies. You can also use them as a reference to write new scripts.

To calculate our metrics, such as normalized rank loss, pearson correlation, and Kruskal-Wallis test, please follow the scripts in `tools/calc_metrics.py`. You can also use it as a reference to calculate metrics for your new environments.

Additionally, we provide our real-to-sim evaluation videos at [this link](TODO).

## Adding New Real-to-Sim Evaluation Environments, Robots, and Policies

Below we provide a step-by-step guide to add new real-to-sim evaluation environments, robots, and policies. 

### Adding New Robots

If you are adding a new robot, perform the following steps:

1. Add your robot urdf to `ManiSkill2_real2sim/mani_skill2/assets/descriptions`. Then, create a test robot script in `tools/robot_object_visualization` to visualize the robot in the SAPIEN viewer. You can use the existing scripts as a reference by changing the loaded urdf path and the initial joint positions (qpos).
   - For SAPIEN viewer control, see [here](#sapien-viewer-controls)
   - For the `visual matching` evaluation setup, you might want to recolor the robot to reduce the real-to-sim perception and evaluation gap. Given a real-world observation image, you can pick a color on the robot arm (a relatively bright color is recommended) using tools like GPick, and then use this color to bucket-paint the robot gripper or the robot arm texture images using tools like GIMP.

2. Create a new robot agent implementation in `ManiSkill2_real2sim/mani_skill2/agents/robots` (which inherit `ManiSkill2_real2sim/mani_skill2/agents/base_agent.py`). You can use the existing robot implementations as a reference.
   - Add a set of controller configurations for the robot arm and the robot gripper. See more controller details in `ManiSkill2_real2sim/mani_skill2/agents/controllers/` and their base class `ManiSkill2_real2sim/mani_skill2/agents/base_controller.py`. You can also add more controller implementations there. 
     - Other relevant functions are the `step_action` function in `ManiSkill2_real2sim/mani_skill2/envs/sapien_env.py` and the `set_action`, `before_simulation_step` functions in `ManiSkill2_real2sim/mani_skill2/agents/base_agent.py`.
   - Add dummy stiffness and damping controller parameters; we will do system identification later.
   - Add cameras to the robot; camera poses are with respect to the link on the robot specified by `actor_uid`. In SAPIEN, camera pose follows ROS convention, i.e., x forward, y left, z up.
     - To minimize the real-to-sim evaluation gap, it is **highly** recommended to calibrate your cameras and input the intrinsic parameters in the robot agent implementation. We found that current policies are quite brittle to camera calibration errors during sim eval.

<details>
<summary>**Notes on camera calibration**: </summary>
Besides using regular camera calibration approaches, you can also use tools like [fSpy](https://github.com/stuffmatic/fSpy). In this case, aim your camera at a large rectangular surface such that all 4 sides of the surface can be seen, and also ensure that 2 vertical lines that are parallel to each other (e.g., 2 lines on the wall) can be seen. Then, input 3 pairs of vanishing lines (2 pairs of lines on the horizontal surface + 1 pair of vertical lines) into fSpy to obtain intrinsic parameters for the camera. For an illustrative example, see `images/fSpy_calibration_example.png`.

Additionally, if you already know the robot joint positions and the camera extrinsics for a particular real-world observation image, you can use `ManiSkill2_real2sim/mani_skill2/examples/demo_manual_control_custom_envs.py` to overlay a real image onto the simulation image and manually tune the intrinsic parameters such that the gripper in the real image aligns with the gripper in the simulation image.
</details>

3. Perform system identification for the robot.
   - First, create a system identification dataset. If you have an existing tensorflow dataset, you can create a subset of trajectories for system identification by modifying `tools/sysid/prepare_sysid_dataset.py`. In other cases, you can create a system identification subset by following the saved pickle file format in `tools/sysid/prepare_sysid_dataset.py` and having the necessary keys in each step of a trajectory.
   - Next, perform system identification using the dataset. You can modify the existing system identification script `tools/sysid/sysid.py`. The script uses simulated annealing algorithm to find better stiffness and damping parameters from the initialization parameters. Examine the system identification logs using `tools/sysid/analyze_sysid_results.py`, and use the best parameters to initialize the next round of system identification with reduced parameter search range. After multiple rounds of system identification, you can then use the best parameters to update the robot agent implementation.

### Adding New Environments

SAPIEN uses an axis convention of x forward, y left, z up for all object and scene assets.

4. Add new object assets to `ManiSkill2_real2sim/data/custom`. 
   - Example object assets are in `ManiSkill2_real2sim/data/custom`. Each object asset contains two components:
     - Visual mesh (a `textured.dae` file combined with its corresponding `.png` texture files, or a single `textured.glb` file)
     - Collision mesh (`collision.obj`). The collision mesh should be watertight and convex. 
   - After adding `collision.obj`, if the collision shape is not yet convex, use `tools/coacd_process_mesh.py` to obtain a convex collision mesh.
   - Use `tools/robot_object_visualization/test_object.py` to visualize the object in the SAPIEN viewer. You can click on an object / object link and press "show / hide" on the right panel to show / hide its collision mesh. 
   - For SAPIEN viewer control, see [here](#sapien-viewer-controls)
   - For the `visual matching` evaluation setup, if you have an asset with good object geometry, then given a real-world observation image, you can use [GeTex](https://github.com/Jiayuan-Gu/GeTex) to bake the real-world object texture into the simulation asset. This is helpful for reducing the real-to-sim evaluation gap.

<details>
<summary>**Notes on modifying and exporting objects from Blender**: </summary>

- If you have an object file opened in Blender, assuming that an object is modeled with y-forward and z-up convention (e.g., the longest side of object is along the +y axis and the shortest side is along the +z axis), you can export the object as `textured.dae` / `textured.glb` for the textured mesh and `collision.obj` for the collision mesh with `x forward, z up` as the output option in Blender. In this case, when you load the object in SAPIEN, the x-axis will become the longest side while the z axis will become the shortest side. This is because the Blender axis convention is y forward, z up, while the SAPIEN axis convention is x forward, z up.
- When modeling objects in Blender, it is recommended to clear its parents (`alt+p > clear parent`) and move the object to the origin (`object > set origin > geometry to origin`). Before exporting objects, set them to have a unit transformation `object > apply > all transforms`. 
- When exporting objects in Blender, you can utilize the "selection only" option to export only the selected objects. Double check that your exported object is correct and is not empty (if it's empty, it means you didn't select the object before exporting it).
- When loading objects in blender, if you are loading `collision.obj` which you saved through the "x forward, z up" convention, then you should also load it using the same axis convention setting. If you loaded `textured.dae` / `textured.glb`, then before modifying it in Blender, first select the object, press `N` to toggle object property channel, set the object rotation to euler mode with all 0s, then start modifying the object, and export objects with the forementioned conventions (x forward, z up).
- For the collision mesh (`collision.obj`), you can further modify it by loading it in Blender and making it (locally) a convex hull (`edit mode > mesh > convex hull`) to reduce the number of vertices and reduce "slipping" behaviors during grasping. You can also use the "decimate" modifier to simplify the collision mesh (`modifier properties > add modifier > decimate > {input your setting and apply}`). For objects like cans, you can also make the bottom of the collision mesh flat (`edit mode > {select desired vertices} > press "s x/y/z 0"`) to reduce wobbly behaviors when the object is placed on a flat surface.
  
</details>

5. Add custom simulation scene backgrounds to `ManiSkill2_real2sim/data/hab2_bench_assets/stages`.
   - In our environments, scene backgrounds are loaded in the `_load_arena_helper` function in `ManiSkill2_real2sim/mani_skill2/envs/custom_scenes/base_env.py`. The existing scenes use the Habitat convention (y-axis up).

<details>
<summary>**Notes on Blender:** </summary>

You can export the `.glb` scenes from Blender. Pay attention to the axis convention. When we modify the ReplicaCAD scenes in Blender, we load and save the scenes using the default Blender axis settings (you can restore these default settings through `Operator Presets > Restore Operator Presets` under the load / save window).
</details>
   
6. If you adopt our visual-matching ("greenscreen") evaluation setup, add the overlay background image (with the robot and interactable objects removed through inpainting) to `ManiSkill2_real2sim/data/real_inpainting`.
   - We use https://cleanup.pictures/ to remove the robot and the interactable objects from the real images.

7. Add new environments to `ManiSkill2_real2sim/mani_skill2/envs/custom_scenes`. You can use the existing environments as a reference. 
   - The environment `reset` function first assesses whether to reconfigure the environment (if so, then we call the `reconfigure` function in `ManiSkill2_real2sim/mani_skill2/envs/sapien_env.py` to load scenes, robot agents, objects, cameras, and lightings). It then calls the `initialize_episode` function to initialize the loaded robots and objects.
   - Our environments load metadata json files for the object assets (in `ManiSkill2_real2sim/data/custom/info_*.json`). Based on your environment implementation, fill in the metadata for each new object asset in existing json files or create new json files.
   - For our existing environments, we implemented the tabletop environments without ray tracing for compatibility with non-RTX GPUs. Though, for Drawer tasks, turning on ray-tracing (`env = gym.make(**kwargs, shadow_dir='rt')`) is necessary as policies heavily rely on brightness contrasts and shadows to infer depth and accomplish the task. If you run ray-tracing environments, they are quite slow on non-RTX GPUs, such as A100.

8. Test your environments using our interactive script `ManiSkill2_real2sim/mani_skill2/examples/demo_manual_control_custom_envs.py`. See the script for more details. In the script, you can manually control the robot and interact with the objects in the environment. You can also invoke the SAPIEN viewer to examine objects and robots. Additionally, for the visual-matching evaluation setup, you can test it to see if the real-world observation image is correctly overlaid onto the simulation observation (e.g., do the table edges align between sim and real). You can then iteratively tune the camera extrinsics and the robot poses to achieve better real-to-sim visual matching.
   - You can set different `env_reset_options` to test different environment configurations.

9. Now we can turn our focus to the policy inference scripts in `./real2sim/`. The main inference script is `real2sim/main_inference.py`, which you can take a look as a reference. Based on your newly-created environments, update the utilities in `real2sim/utils/env/env_builder.py` and `real2sim/utils/env/observation_utils.py`.

10. If your policy is already implemented in our repo (i.e., RT-* and Octo), you can now perform policy evaluations in simulation! This is done through the policy inference scripts in `scripts/`. You can use the existing scripts as a reference to write new scripts. After running sim eval, modify the scripts in `tools/calc_metrics.py` to calculate the metrics in your newly-created environments, such as normalized rank loss, pearson correlation, and Kruskal-Wallis test.
   

### Adding New Policies

11.  Implement new policy inference scripts in `real2sim/{your_new_policy}`, following the examples for RT-1 and Octo policies. 

12.  Modify `real2sim/main_inference.py`, along with the utilities in `real2sim/utils/env/env_builder.py` and `real2sim/utils/env/observation_utils.py` to support your new policies.

13.  Add policy inference scripts in `scripts/` to perform real-to-sim evaluation. Then, modify the scripts in `tools/calc_metrics.py` to calculate the metrics for your new policies.





### Debugging Scripts and Other Tools

To visualize robots and objects, see `tools/robot_object_visualization`.

To debug robot-object interactions in an environment along with real-to-sim visual matching, see `ManiSkill2_real2sim/mani_skill2/examples/demo_manual_control_custom_envs.py`.

We have also provided helpful example debugging tools in `real2sim/utils/debug` to help you debug your new robots and policies. Specifically,
- `real2sim/utils/debug/{robot_name}_test_dataset_inference_rollout_gt_traj_in_sim.py` steps ground-truth actions in a demonstration dataset in an open-loop manner and records the resulting observation video and robot qpos. This is helpful for visualizing the real-to-sim control gap after system identification.
- `real2sim/utils/debug/inference_real_video.py` feeds a sequence of (real evaluation) video frames into the policy and executes the resulting policy actions, which is helpful for debugging whether the behaviors of implemented policies are correct and reasonable. It can also feed an inpainting image with a robot arm rendered in simulation to the policy and sequentially execute the policy actions, which is helpful for investigating the effect of robot arm / gripper textures on the real-to-sim evaluation gap.
- `real2sim/utils/debug/rt1_plot_dataset_inference_trajectory.py` plots ground-truth and policy-predicted demonstration action trajectories in a dataset, which is helpful for debugging whether the behaviors of implemented policies are correct and reasonable.

We also provide some visualization scripts:
- `tools/merge_videos.py` allows you to merge the simulation evaluation videos into a single video.
- `tools/save_video_frame.py` saves a particular frame within a video.




## Appendix

#### SAPIEN viewer controls

- In the viewer, click on an object / articulated object link and press "f" to focus on it. 
- Use right mouse button to rotate; middle-mouse-button + shift to translate. Scroll the middle mouse button to zoom in / out (if you press shift when you scroll, you can zoom in / out slower).
- Under "scene hierarchy" on the bottom left, you can select different actors and articulation links based on their names. 
- When an articulated object is selected (e.g., robot), then under "articulation" on the bottom right, you can move the scrollbar to change each of its joint positions / angles. 
- Press "pause" on the top left to pause the simulation. 
- Press "g" to grab object; "g" + "x"/"y"/"z" to move object along x/y/z axis.

#### Asset missing errors and MS2_ASSET_DIR environment variable

If you are not running scripts under the `ManiSkill2_real2sim` directory, but in the script, you need to create a ManiSkill2 environment, then some asset missing errors might be reported. In this case, please make sure this environment variable is set:

```
export MS2_ASSET_DIR={path_to_ManiSkill2_real2sim}/data
```

#### Other troubleshooting tips

- `[error] Failed to cook a mesh from file: {path}/collision.obj`: this error is most likely caused by the collision mesh not being watertight or not being convex. You can use `tools/coacd_process_mesh.py` to obtain a convex collision mesh.

#### More notes

- The real-world Google Robot controller is non-blocking. For simplicity, currently we implement 3hz fixed frequency control (following RT-1 paper) in simulation.
- We use `tcp` to refer to the tool center point of the robot end-effector throughout the codebase.


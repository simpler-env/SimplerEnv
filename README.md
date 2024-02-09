# Real2Sim

### Installation

```
sudo apt install ffmpeg
```

Create an anaconda environment: 
```
conda create -n real2sim python=3.9
```

Clone this repo:
```
git clone https://github.com/xuanlinli17/real2sim --recurse-submodules
cd robotics_transformer
git-lfs pull
```

Install ManiSkill2:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .

wget https://dl.fbaipublicfiles.com/habitat/ReplicaCAD/hab2_bench_assets.zip -P data
cd data && unzip -q hab2_bench_assets.zip -d hab2_bench_assets
rm hab2_bench_assets.zip

cd ..
python -m mani_skill2.utils.download_asset ycb # choose no when prompted to remove ycb
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
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .
unzip rt_1_x_tf_trained_for_002272480_step.zip
rm rt_1_x_tf_trained_for_002272480_step.zip
```

Octo:
```
cd {this_repo}/octo
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -e .
```

**SysID**
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```

**Export custom model**
```
Blender axis convention is y forward, z up; SAPIEN axis convention is x forward, z up. If you manually edit or create objects in Blender, assuming that an object is modeled with y-forward and z-up convention, you can export the object as `textured.dae` for the textured mesh and `collision.obj` for the collision mesh with `x forward, z up` as the output option in Blender. Then, use `tools/coacd_process_mesh.py` to obtain a convex collision mesh.

Make mesh a (local) convex hull to reduce "slipping" behaviors
```


**Scripts**

See `scripts/rt1_pick_coke_can_eval.sh`

**Evaluation Note**

When you evaluate "pick coke can", some cases might show success in our automated metric (since the coke can leaves the table), but the lifting distance is too small (<2cm) and insignificant. In this case I'll manually check these videos, and if the coke can is too close to the table, I'll count them as failure.




real2sim/main_inference.py
real2sim/utils/env/env_builder.py
real2sim/utils/env/observation_utils.py
real2sim/tools/sysid/sysid.py

tcp = tool center point of end-effector
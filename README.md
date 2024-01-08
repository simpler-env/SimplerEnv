# Real2Sim

### Installation

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



Install tensor2robot and roboticstransformer:
```
cd {this_repo}/tensor2robot
pip install -r requirements.txt
cd proto
protoc -I=./ --python_out=`pwd` t2r.proto

cd {this_repo}/robotics_transformer
pip install -r requirements.txt
```

Install this package:
```
cd {this_repo}
pip install -e .
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

<!-- Install latest torch (>=2.2.1):
```
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
```

```
pip install git+https://github.com/pytorch-labs/segment-anything-fast.git
``` -->

**SysID**
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```

**Export custom model**
```
test_object.py: x axis should be point towards the can symbol; y axis should point up

Blender export: Blender uses y forward, z up; SAPIEN: x forward, z up
Collada and obj need to export w/ the same forward & up axis to ensure right collision

Make mesh a (local) convex hull to reduce "slipping" behaviors
```


**Reconstructing mesh from multi-view images**

Install `segment_anything`

<!-- ```
git clone https://github.com/bennyguo/instant-nsr-pl
pip install torch_efficient_distloss nerfacc==0.3.3 PyMCubes omegaconf pyransac3d
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
``` -->

Install `instant-nsr-pl`

```
git clone https://github.com/bennyguo/instant-nsr-pl
conda create -n instant_nsr_pl python=3.9
conda activate instant_nsr_pl
cd {this_repo}/instant-nsr-pl
# Attention: torch must match your local cuda version; the following command assumes you have cuda11.8
pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```


```
conda activate real2sim
video_dir=/hdd/object_videos/coke_can/

# If you have a video,
python tools/mesh_reconstruction/video_to_image.py --folder ${video_dir} --video-frame -1 --skip-frame 5
# If you have a series of images,
python tools/mesh_reconstruction/rename_images.py --folder ${video_dir}/extracted_images

# Then,
python tools/mesh_reconstruction/generate_seg_mask.py --input-dir ${video_dir}/extracted_images/ \
  --output-dir ${video_dir} --prompt "coke can" --export-format colmap
python tools/mesh_reconstruction/remove_image_floaters.py --folder ${video_dir}/colmap/ --threshold 8000

cd {this_repo}/instant-nsr-pl
python scripts/imgs2poses.py ${video_dir}/colmap --masks-path ${video_dir}/colmap/masks

conda activate instant_nsr_pl
python launch.py --config configs/nerf-colmap.yaml --gpu 0 --train dataset.root_dir=/hdd/xuanlin/object_videos/coke_can/colmap/ \
    dataset.apply_mask=True dataset.load_data_on_gpu=True dataset.img_downscale=5 tag=coke_can_1 model.radius=1.0 trainer.max_steps=20000
# train NeuS with mask
python launch.py --config configs/neus-colmap.yaml --gpu 0 --train dataset.root_dir=/hdd/xuanlin/object_videos/coke_can/colmap/ \
    dataset.apply_mask=True dataset.load_data_on_gpu=True dataset.img_downscale=5 tag=coke_can_1 system.loss.lambda_mask=0.1 model.radius=1.0
```


**TODO**

```
cd {this_repo}
git clone https://github.com/SarahWeiii/TensoRF
git clone https://github.com/SarahWeiii/diffrec_ddmc
pip install git+https://github.com/NVlabs/nvdiffrast/

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit # latest torch version
# if not already, update gcc & g++ version to 8+ (and make sure they have the same version)
pip install --global-option="--no-networks" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch
imageio_download_bin freeimage

git clone https://github.com/SarahWeiii/diff_dmc.git
cd diff_dmc/dmc_cuda
pip install -e .

cd {this_repo}
```


```
video_dir=/hdd/object_videos/coke_can/
python tools/mesh_reconstruction/generate_seg_mask.py --input-dir ${video_dir}/extracted_images/ \
  --output-dir ${video_dir} --prompt "coke can" --export-format nerf_synthetic
cd ${video_dir}/nerf_synthetic/seg_image
python {this_repo}/TensoRF/dataLoader/colmap2nerf.py --run_colmap \
  --images ./train --masks ../seg --out ./transforms_train.json  --colmap_matcher exhaustive  --aabb_scale 1
python {this_repo}/TensoRF/dataLoader/colmap2nerf.py \
  --images ./test --masks ../seg --out ./transforms_test.json  --colmap_matcher exhaustive  --aabb_scale 1

# Train TensoRF
cd {this_repo}/TensoRF
python train.py --config configs/lego.txt --datadir /hdd/xuanlin/object_videos/coke_can/nerf_synthetic/seg_image/ --dataset_name own_data \
  --expname tensorf_coke_VM_t12_6_2_mlp3x64 --data_dim_color 12 --featureC 64 --view_pe 6 --fea_pe 2 --batch_size 4096

cd {this_repo}/diffrec_ddmc
python train.py --config configs/nerf.txt --ref_mesh /hdd/xuanlin/object_videos/coke_can/nerf_synthetic/seg_image/ --base_model /home/xuanlin/Real2Sim/TensoRF/log/tensorf_coke_VM_t12_6_2_mlp3x64/tensorf_coke_VM_t12_6_2_mlp3x64.th --mesh_scale 3.0 --out_dir nerf_coke_tensorVM_preload_10k_t12_6_2_3x64_dmc256_wd --tex_dim 12 --feape 2 --viewpe 6 --display "all_tex" "gb_pos" "wo" --iter 10000 --batch 1 --learning_rate_1 0.0005 --learning_rate_2 0.0005 --lock_pos 0 --model_type pretrained_tensorf_rast --tex_type tensorVM_preload --dmtet_grid 256 --lr_decay 0.1 --sdf_grad_weight 0 --sdf_sparse_weight 0 --shader_internal_dims 64 --normal_smooth_weight 0 --multires 0

```



```
python tools/mesh_reconstruction/video_to_image.py --folder /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke --video-frame 0
python tools/mesh_reconstruction/video_to_image.py --folder /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke_bottom --video-frame 0

python tools/mesh_reconstruction/generate_seg_mask.py --input-dir /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke --prompt "coke can"
python tools/mesh_reconstruction/generate_seg_mask.py --input-dir /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke_bottom --prompt "coke can"

colmap model_converter --input_path /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/calibration/sparse/0 \
  --output_path /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/calibration/sparse/0 --output_type TXT

cd /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image
python {this_repo}/TensoRF/dataLoader/colmap2nerf.py \
  --text /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/calibration/sparse/0 \
  --images /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/train \
  --out /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/transforms_train.json

cd /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image
python {this_repo}/TensoRF/dataLoader/colmap2nerf.py \
  --text /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/calibration/sparse/0 \
  --images /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/test \
  --out /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/transforms_test.json

# Train TensoRF
cd {this_repo}/TensoRF
python train.py --config configs/lego.txt --datadir /hdd/xuanlin/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/ --dataset_name own_data \
  --expname tensorf_coke_VM_t12_6_2_mlp3x64 --data_dim_color 12 --featureC 64 --view_pe 6 --fea_pe 2 --batch_size 40960

python train.py --config configs/lego.txt --datadir /hdd/xuanlin/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/ --dataset_name own_data \
  --expname tensorf_coke_VM_t12_6_2_mlp3x64 --data_dim_color 12 --featureC 64 --view_pe 6 --fea_pe 2 --ckpt log/tensorf_coke_VM_t12_6_2_mlp3x64/tensorf_coke_VM_t12_6_2_mlp3x64.th --render_only 1 --render_surface 1

# scp -r xuanlin@minkowski:/home/xuanlin/Real2Sim/TensoRF/log/tensorf_coke_VM_t12_6_2_mlp3x64 ~/Downloads

# Train diffrec_ddmc
cd {this_repo}/diffrec_ddmc
python train.py --config configs/nerf.txt --ref_mesh /hdd/xuanlin/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/ --base_model /home/xuanlin/Real2Sim/TensoRF/log/tensorf_coke_VM_t12_6_2_mlp3x64/tensorf_coke_VM_t12_6_2_mlp3x64.th --mesh_scale 3.0 --out_dir nerf_coke_tensorVM_preload_10k_t12_6_2_3x64_dmc256_wd --tex_dim 12 --feape 2 --viewpe 6 --display "all_tex" "gb_pos" "wo" --iter 10000 --batch 2 --learning_rate_1 0.001 --learning_rate_2 0.001 --lock_pos 0 --model_type pretrained_tensorf_rast --tex_type tensorVM_preload --dmtet_grid 256 --lr_decay 0.1 --sdf_grad_weight 0 --sdf_sparse_weight 0 --shader_internal_dims 64 --normal_smooth_weight 0 --multires 0

# scp -r xuanlin@minkowski:/home/xuanlin/Real2Sim/diffrec_ddmc/out/nerf_coke_tensorVM_preload_10k_t12_6_2_3x64_dmc256_wd ~/Downloads
```
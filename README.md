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
python -m mani_skill2.utils.download_asset ycb
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


**Reconstructing mesh from multi-view images**

Install `segment_anything`

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
python train.py --config configs/lego.txt --datadir /hdd/lightstage/lightstage.nrp-nautilus.io/xuanlin/coke/seg_image/ --dataset_name own_data \
  --expname tensorf_coke_VM_t12_6_2_mlp3x64 --data_dim_color 12 --featureC 64 --view_pe 6 --fea_pe 2
```
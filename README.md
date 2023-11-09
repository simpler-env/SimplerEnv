# Real2Sim

### Installation

Create an anaconda environment: 
```
conda create -n real2sim python=3.9
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
pip install -e .
```

Download RT-1 Checkpoint:
```
cd {this_repo}
gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .
unzip rt_1_x_tf_trained_for_002272480_step.zip
rm rt_1_x_tf_trained_for_002272480_step.zip
```
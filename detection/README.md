# COCO object detection and instance segmentation

PS: based on the [ReviewKD's codebase](https://github.com/dvlab-research/ReviewKD).

## Environment

* 4 GPUs
* python 3.6
* torch 1.9.0
* torchvision 0.10.0

## Installation

Our code is based on Detectron2, please install Detectron2 refer to https://github.com/facebookresearch/detectron2.

Please put the [COCO](https://cocodataset.org/#download) dataset in datasets/.

Please put the pretrained weights for teacher and student in pretrained/. You can find the pretrained weights [here](https://github.com/dvlab-research/ReviewKD/releases/). The pretrained models we provided contains both teacher's and student's weights. The teacher's weights come from Detectron2's pretrained detector. The student's weights are ImageNet pretrained weights.

## Training

```
# Tea: R-101, Stu: R-18
python3 train_net.py --config-file configs/DKD/DKD-R18-R101.yaml --num-gpus 4

# Tea: R-101, Stu: R-50
python3 train_net.py --config-file configs/DKD/DKD-R50-R101.yaml --num-gpus 4

# Tea: R-50, Stu: MV2
python3 train_net.py --config-file configs/DKD/DKD-MV2-R50.yaml --num-gpus 4

```

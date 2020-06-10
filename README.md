# Siamese Networks with Hinge Loss for Real-Time Visual Tracking

## Introduction

Siamese networks are really popular in the field of visual tracking because of their balanced efficiency, accuracy and speed. But, the backbone network utilized in these trackers is still the classical AlexNet, which does not capture the capabilities of modern deep neural networks.

Our proposal for improve the [SiamDW](https://arxiv.org/pdf/1901.01660v3.pdf) performances of fully convolutional siamese trackers by,

1. Using Hinge loss function to improve the performance of SiamDW implementation

<div align="center">
  <img src="demo/vis.gif" width="800px" />
  <!-- <p>Example SiamFC, SiamRPN and SiamMask outputs.</p> -->
</div>

<!-- :tada::tada: **Highlight !!**
Siamese tracker is severely sensitive to hyper-parameter, which is a common sense in tracking field. Although significant progresses have been made in some works, the result is hard to reproduce. In this case, we provide a [parameter tuning toolkit]() to make our model being reproduced easily. We hope our efforts and supplies will be helpful to your work. -->

## Main Results

Results are based on the CIRResNet22-RPN model by using a simpler loss function (logistic loss), vs our Hinge loss function

### Main results on VOT and OTB

| Models            | VOT16 | VOT17 | VOT18 |
| :---------------- | :---: | :---: | :---: |
| Logistic Loss     | 0.331 | 0.376 | 0.294 |
| Hinge Loss (ours) | 0.312 | 0.318 | 0.322 |
| Focal Loss (ours) |  ---  |  ---  |  ---  |

### Environment

Initially encironment: GPU: NVIDIA .GTX 1050
Advanced encironment: The code is developed with Ryzen 5 1600x 6 core 12 thread CPU @ 4.20GHz RAM: 16GB GPU: NVIDIA .RTX2060

# Training

## Data preparation

- There are preprocessed datasets `VID`, `YTB`, `GOT10K`, `COCO`, `DET` and `LASOT`. You can download it from [GoogleDrive](https://drive.google.com/drive/folders/1_A54r4QMRVn4oYfGXIjtY5e7o8TeEnsJ).
- You might have limitations due to the GoogleDrive download capacity.
- If you face up with download limit exceeded issue, you can copy the files into your GoogleDrive.

## Pretrained model preparation

The code will download pretrained model from GoogleDrive automatically. If failed, please download from [GoogleDrive](https://drive.google.com/open?id=1RIMB9542xXp60bZwndTvmIt2jogxAIX3) , and put them to `pretrain` directory.

## Conda preparation

```
sh install_rpn.sh
```

## Setting preparation

Modify yaml files in `experiment/train/` according to your needs.

## One-key Running

This script will excute `train`-`epoch-test`-`hyper-parameter tuning` automatically to save your time.

```
python siamese_tracking/onekey.py
```

## Data optimization

- Different training data and mix-up ratio will affect final performance. You can modify `WITCH_USE` in yaml files of `experiment/train/` to find witch data is better for your task. Also, modify `USE` in yaml files to try different mix-up ratio. High quality training data is beneficial to training. We used ["COCO"] dataset which is a large-scale object detection work.

## Backbone optimization

We provide `ResNet`, `Inception`, `DenseNet`, `NasNet` and `ResNext` in codes. <br/>
Add your backbone in `lib/models/backbone.py`. Pretraining backbone on Imagenet is always good to training.

## Loss optimization

We changed loss function from Logistic Loss into Hinge loss. <br/>

Additionally, you can add your loss function in `lib/models/siamfc.py` or `lib/models/siamrpn.py`

<br/>

# Test

- Download models from [GoogleDrive](https://drive.google.com/drive/folders/19dBWxOqZnvM0FsgXGzH2Y7Bg7wgYMEoO?usp=sharing) , and put them to `snapshot` directory

## Test on a specific video

eg,

```
python siamese_tracking/run_video.py --arch SiamRPNRes22 --resume snapshot/CIResNet22_RPN.pth --video videos/bag.mp4
```

- The opencv version here is 4.1.0.25, and older versions may be not friendly to some functions.
- If you try to conduct this project on a specific tracking task, eg. pedestrian tracking, it's suggested that you can tuning hyper-parameters on your collected data with our tuning toolkit detailed below.

## Test through webcam

eg,

```
python siamese_tracking/run_webcam.py --arch SiamRPNRes22 --resume snapshot/CIResNet22_RPN.pth
```

- The opencv version here is 4.1.0.25, and older versions may be not friendly to some functions.
- You can embed any tracker for fun. This is also a good way to design experiments to determine how environmental factors affect your tracker.

## Test on benchmarks

### Data preparation

The test dataset [VOT](https://votchallenge.net/) should be arranged in `dataset` directory. Your directory tree should look like this:

```
${Tracking_ROOT}
|—— experimnets
|—— lib
|—— snapshot
|—— dataset
  |—— VOT2015
     | —— videos...
  |—— VOT2016
     | —— videos...
  |—— VOT2017
     | —— videos...
|—— run_tracker.py
|—— ...

```

### Conda preparation

```
sh install_rpn.sh
```

### Toolkit preparation

- Set up vot-toolkit according to official [tutorial](http://www.votchallenge.net/howto/integration_channels.html)
- Modify `path_to/toolkit` in `lib/core/get_eao.m` to your vot-toolkit path
- In your matlab install path (MATLAB2017b or higher),

```
cd $matlab_path/R2018b/extern/engines/python
python setup.py install
```

- Download datasets VOT2015, VOT2016, and VOT2017 and put them into the dataset directory.

### Run tracker

```bash
CUDA_VISIBLE_DEVICES=0 python ./siamese_tracking/test_siamrpn.py --arch SiamRPNRes22 --resume ./snapshot/CIResNet22_RPN.pth --dataset VOT2015 --cls_type thinner
or
CUDA_VISIBLE_DEVICES=0 python ./siamese_tracking/test_siamrpn.py --arch SiamRPNRes22 --resume ./snapshot/CIResNet22_RPN.pth --dataset VOT2016 --cls_type thinner
or
CUDA_VISIBLE_DEVICES=0 python ./siamese_tracking/test_siamrpn.py --arch SiamRPNRes22 --resume ./snapshot/CIResNet22_RPN.pth --dataset VOT2017 --cls_type thinner
```

### Analysz testing results

We implemented out VOT benchmark to estimate Distance Precision, Average Overlap Ratio, and Average Center Location error. You can get the evaluation results as in the following.

- VOT

```bash
python ./evalutionVOT.py VOT2015
or
python ./evalutionVOT.py VOT2016
or
python ./evalutionVOT.py VOT2017
```

### Attention !!

- Recently we found that the image is slightly inconsistent while using different OpenCV version. And the speed of some opencv versions are relatively slow for some reason. It is recommended that you install packages above.
- The SiamRPN based model is trained on pytorch0.4.1, since we found that memory leak happens while testing SiamRPN on pytorch0.3.1 with multithread tools.

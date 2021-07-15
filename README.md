# Pytorch - Retina Face Detector

<a href="https://wandb.ai/nmd2000/Retina-Face/"><img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg" alt="Visualize in WB"></a>

An implementation of **RetinaFace: Single-stage Dense Face Localisation in the Wild**  [[1]](#1) by [Pytorch](https://pytorch.org/). The model achieved 68.15% average precision in WIDER FACE [[2]](#2) hard set (on validation set) where I made few changes in its anchors box generative methods, its multi-task loss function and RetinaFace architecture (will be discuss in section below). 

## Table of contents
- [RetinaFace](#retinaface)
- [Result](#result)
- [Usage](#usage)
- [Reference](#reference)

# RetinaFace
<div style="text-align:center"><img src="/report/images/retina-model.jpg" /></div>

**Feature Pyramid.** We used Feature pyramid [[3]](#3) on Resnet18 as the backbone of the model in order to extract multi-scale of image features. On the Object Detection task, the problem of detecting small object was a remarkable hard. The Feature pyramid computed a feature hierarchy layer by layer and produces feature maps of different spatial resolutions (multiple levels) as output where the high-resolution with weak feature can combine with strong low-resolution feature. Therefore, Feature pyramid can extremely increase semantic context value.

<div style="text-align:center"><img src="/report/images/fpn.png" /></div>

***Figure 1.** Top: a top-down architecture with skip connections, where predictions are made on the finest level. Bottom: our model that has a similar structure but leverages it as a feature pyramid, with predictions made independently at all levels. [[3]](#3)*

**Context Module.** The original design of the *Context Module* was inspired by Single Stage Headless Detector [[4]](#4) to enhance the model's contextual reasoning power for capturing tiny faces. My RetinaFace model only use 1 Context Module shared for all feature pyramid output (while in original model are 5 seperate Context Module).

<div style="text-align:center"><img src="/report/images/context-module.png" /></div>

***Figure 2**. SSH context module [[4]](#4)*

**Anchor Settings.** When I took a quick survey on WIDER FACE dataset [[2]](#2), it's shocking to found that number of samples which have *width* and *height* in between [1, 25] take ~71% and the most common bounding box aspect ratio are 1:2 and 1:1. Therefore, I made a few change in the generative method of the anchor box in order to strength the capability of detecting smail face, detail in table below:

| Feature Pyramid  | Stride |        Anchor       | Number of anchor on feature map |
|:----------------:|:------:|:-------------------:|:-------------------------------:|
| P2 (120×120×64)  |    4   |   8, 10.08, 12.67   |              86400              |
| P3 (60×60×64)    |    8   |   16, 20.16, 25.40  |              21600              |
| P4 (30×30×64)    |   16   |   32, 40.32, 50.80  |               5400              |
| P5 (15×15×64)    |   32   |  64, 80.63, 101.59  |               1350              |
| C6/P6 (8×8×64)   |   64   | 128, 161.26, 203.19 |               384               |

The bounding box are varible size from 8×8 up to 203×203 (this can be a minus because the model is not able to detect face have size bigger than 203×203 in 480×480 image) and 75% anchors box were came from P2.

*Note:* In pratice, I once set number of anchor on P2 up to ~172k boxes. However, I noticed that the model was fail with huge number of false positive prediction.

**Loss function.** Coming soon.

# Result
Due to the limitation of resource, the training process was trained on Colab Free GPU (with 15GB Vram). I have to resize the image into 480×480 (to advoid OOM) and set number of output channel on each feature map to 64 while it's 256 in original model and 640×640 as input shape. I used SGD optimizer with momentum = 0.9, weight decay = 5e-4, batch-size is 8 and learning rate starts from 1e-3 and decrease 0.7 time after 5 epochs. The process terminated after 30 epochs and the result are shown below:

|                   Backbone                   | EASY SET | MEDIUM SET | HARD SET |
|:--------------------------------------------:|:--------:|:----------:|:--------:|
| MobileNet0.25 (original image scale)         |  90.70%  |   88.16%   |  73.82%  |
| MobileNet0.25 (same parameter with Mxnet)    |  88.67%  |   87.09%   |  80.99%  |
| Resnet151 (Mxnet)                            |  96.942% |   96.175%  |  91.857% |
| Resnet18 (our)                               |  74.35%  |   71.65%   |  68.15%  |

*Notes:* You can find our (pre-processing) dataset and see training experiment through [Weight&Bias](https://wandb.ai/nmd2000/Retina-Face/).

# Training
## Select model
While my MobileNet.25 doesn't have any pre-train weight, I encourage to use a pre-trained Resnet (18, 34, 50, 152) by calling e.g. `--model resnet18` when training

## Trigger training
Train a RetinaFace on WIDER FACE by specifying batch-size, pretained weight, epoch, trigger download pre-processed dataset from Wandb database, ...

```
# train RetinaFace on WIDER FACE with resnet18 as backbone
$ python train.py --image_size 640 --batchsize 8 --epochs 5 --model resnet18 --download --device 0
                                   --batchsize 16           --model resnet34
                                   --batchsize 32           --model resnet50
```


# Reference
<a id="1">[1]</a> Jiankang Deng **andothers**. “Retinaface: Single-stage dense face localisation in the wild”.**in**: arXiv preprint arXiv:1905.00641 (2019).

<a id="2">[2]</a> Shuo Yang **andothers**. “Wider face: A face detection benchmark”. **in:** *Proceedings of the IEEE conference on computer vision and pattern recognition.* 2016, **pages** 5525–5533.

<a id="3">[3]</a> Tsung-Yi Lin **andothers**. “Feature pyramid networks for object detection”. in: *Proceedings of the IEEE conference on computer vision and pattern recognition.* 2017, **pages** 2117–2125

<a id="4">[4]</a> Mahyar Najibi **andothers**. “Ssh: Single stage headless face detector”. **in:** *Proceedings of the IEEE international conference on computer vision.* 2017, **pages** 4875–4884.


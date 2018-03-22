# Object Detection from Video Tubelets with Convolutional Neural Networks
[arXiv](https://arxiv.org/abs/1604.04053)
[TOC]
## Introduction
1. existing video detection: mainly focus on detecting one specific class of objects, such as pedestrians [31], cars [19], or humans with actions [13, 11]
> 现有的视频检测主要针对某一类物体

2. 直接检测跟踪框的效果不好（不明白怎么检测跟踪结果），还不如object proposals。因为检测对位置敏感，而且track和object proposals有误匹配

3. propose a multi-stage framework based on deep CNN detection and tracking for object detection
   1. a **tubelet proposal module** that combines object detection and object tracking for tubelet object proposal
   2. a **tubelet classification and re-scoring module** that performs spatial max-pooling for robust box scoring and temporal convolution for incorporating temporal consistency
> 提出了基于跟踪和检测的多阶段框架来做检测。1、tubelet提出，2、tubelet分类和得分调整

## Contributions
1. multi-stage framework is proposed for object detection in videos
2. 对跟踪和检测对视频检测任务的影响进行了仔细讨论
3. special temporal convolutional neural network is proposed to incorporate temporal information into object detection from video. 整合了时空信息

## Method
### Framework
![videoTubelets](./.assets/videoTubelets.jpg)

### Spatio-temporal tubelet proposal
> 1. 简单的图像检测导致检测得分波动( the detection scores on consecutive frames usually have large fluctuations)
> 2. 简单的跟踪导致漂移(usually tends to drift due to large object appearance changes)

1. Image object proposal with the Selective Search [34]
2. Object proposal scoring
3. High-confidence proposal tracking
> 这部分基本同T-CNN

### Tubelet classification and rescoring
> 直接分类tubelets box的效果不如静态检测RCNN，原因有：1. tubelets box比SS box小。2. 检测器对微小的位置偏移敏感。3. 跟踪过程中，有减少冗余，所以tubelets box比SS box稀疏，导致NMS不能正常工作。4. 检测得分在tubelets box上较大变化

1. Tubelet box perturbation and max-pooling: to replace tubelet boxs with boxes of higher confidence
   1. generate new boxes around each tubelet box on each frame by randomly perturbing the boundaries of the tubelet box
   2. replace each tubelet box with original object detections that have overlaps with the tubelet box beyond a threshold
> 对每个tubelets box做扩增，然后保留得分最高的一个

2. Temporal convolution and re-scoring
> 经过上面的过程后，检测得分依然波动

![TCN](./.assets/TCN.jpg)
train a class-specific TCN
$$
\begin{array}c
\text{detection scores, tracking scores and anchor offsets} \\
\downarrow \\
\text{probablities whether each tubelet box contains objects of the class}
\end{array}
$$
## References
### tracking-by-detection
[1] M. Andriluka, S. Roth, and B. Schiele. People-tracking-by-detection and people-detection-by-tracking. CVPR, 2008.
[2] A. Andriyenko, K. Schindler, and S. Roth. Discretecontinuous optimization for multi-target tracking. CVPR, 2012.
[3] S.-H. Bae and K.-J. Yoon. Robust online multi-object tracking based on tracklet confidence and online discriminative appearance learning. CVPR, 2014.
[26] H. Possegger, T. Mauthner, P. M. Roth, and H. Bischof. Occlusion geodesics for online multi-object tracking. CVPR, 2014.

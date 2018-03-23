# Single-Shot Refinement Neural Network for Object Detection
[arXiv](https://arxiv.org/abs/1711.06897)
[Caffe](https://github.com/sfzhang15/RefineDet)
[TOC]
## Introduction
1. advantage of two-stage methods
   1. using two-stage structure with sampling heuristics to handle class imbalance
   2. using two-step cascade to regress the object box parameters
   3. using two-stage features to describe the objects

## Architecture
### overview
   1. anchor refinement module (ARM)
      1. remove negative anchors so as to reduce search space for the classifier
      2. coarsely adjust the locations and sizes of anchors to provide better initialization for the subsequent regressor
      > 移除negative anchor来减小分类的搜索空间，粗略调整大小和位置

   2. object detection module (ODM)
   > 回归和分类on the refined anchors

### Components
   1. transfer connection block (TCB), converting the features from the ARM to the ODM for detection
   2. two-step cascaded regression, accurately regressing the locations and sizes of objects
   3. negative anchor filtering, early rejecting well-classified negative anchors and mitigate the imbalance issue

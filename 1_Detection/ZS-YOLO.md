# Zero-Shot Detection
[arXiv](https://arxiv.org/abs/1803.07113)
[TOC]
## Introduction
1. 动机：检测未标注的object
2. 方法：训练时同时整合语义属性和视觉特征

## Approah
![ZSYOLO](./.assets/ZSYOLO.jpg)
1. Architecture
   1. feature extraction module
   2. object localization module
   3. **semantic prediction module**
      1. For each testing bounding box proposal we can obtain its semantic representation
      > 获得语义表达

      2. As ZS-YOLO is trained end-to-end, the loss of this layer will back propagate so that the learned visual representations will also be influenced by similarities in the semantic domain.
      > 视觉特征同时被语义相似性所影响

   4. objectiveness confidence prediction module
      1. 利用语义信息+视觉信息+位置信息来分类

2. Loss
   1. Object Localization Loss
   2. Semantic Loss: 学习一个余弦相似度
   $$ L_{attr}=\sum_k[\lambda_{obj}\mathbb I_k^{obj}(S(\hat y_k,y_k)-1)^2+\lambda_{noobj}\mathbb I_k^{noobj}(\max_{c\in C_{seen}} S(\hat y_k,y_c)-0)^2] $$
   $S=\frac{\vec{a}\cdot \vec{b}}{||a||||b||}$ 余弦相似度(cosine similarity )
   $\lambda_{obj}=5,\lambda_{noobj}=1$用于平衡前景背景的不平衡
   $\mathbb I_k^{noobj}=1$ 当且仅当第$j$个anchor与gt box的overlap=0
   $\mathbb I_k^{obj}=1$ 当且仅当(if and only if)
      + 第$k$个box由第$j$个anchor预测
      + gt box的center落入第$j$个anchor(falls into cell $j$)
      + 在第$j$个anchor预测的5个box中，第$k$个box与gt box的overplap最大
3. Confidence Loss

## learned
重点在semantic prediction module，实际上就是学习一个余弦相似度，object的gt余弦相似度为1，背景为0

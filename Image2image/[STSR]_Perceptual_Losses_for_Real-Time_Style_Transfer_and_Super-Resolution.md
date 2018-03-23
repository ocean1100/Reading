# Perceptual Losses for Real-Time Style Transfer and Super-Resolution
[arXiv](https://arxiv.org/abs/1603.08155)

<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](#perceptual-losses-for-real-time-style-transfer-and-super-resolution)
  * [Introduction](#introduction)
  * [Method](#method)

<!-- tocstop -->

## Introduction
1. per-pixel losses used by these methods do not capture perceptual differences between output and ground-truth images. For example, consider two per-pixel losses used by these methods do not capture perceptual differences between output and ground-truth images. For example, consider two
2. Contribution
rather than using per-pixel loss functions depending only on low-level pixel information, we train our networks using **perceptual loss** functions that depend on high-level features from a pretrained loss network.

## Method
![framework](./.assets/framework.jpg)

1. Feature Reconstruction Loss
$$ \mathcal l_{frat}^{\phi,j}(\hat y,y) = \frac{1}{C_jH_jW_j}||\phi(\hat y)-\phi(y)||^2_2
$$
2. Total Variation Regularization
To encourage spatial smoothness in the output image,  make use of total variation regularizer $\mathcal l_{TV}(\hat y)$.

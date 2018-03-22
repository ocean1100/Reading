# Object Detection in Videos with Tubelet Proposal Networks
[arXiv](https://arxiv.org/abs/1702.06355)
[TOC]
## Introduction
1. tubelets: Similar to the bounding box proposals in the static object detection, the counterpart in videos are called tubelets, which are essentially sequences of bounding boxes proposals.
2. The ideal tubelets for video object detection should be long enough to incorporate temporal information while diverse enough to ensure high recall rates

## Tubelet proposal networks
1. Architecture
![TPN](./.assets/TPN.jpg)

2. spatial anchors
$b_t^i=(x_t^i, y_t^i, w_t^i, h_t^i)--$ the $i$th static box proposal at time $t$
$b_1^i --$ spatial anchors

2. Regression $R()$ for estimating the relative movements w.r.t. the spatial anchors
$$
\begin{array}l
m_1^i,...,m_\omega^i = R(r_1^i,...,r_\omega^i) \\
\Delta x_t^i = (x_t^i - x_1^i)/w_1^i \\
\Delta y_t^i = (y_t^i - y_1^i)/h_1^i \\
\Delta w_t^i = \log(w_t^i / w_1^i) \\
\Delta h_t^i = \log(h_t^i / h_1^i) \\
\end{array}
$$
> $r$ visual features , $\omega$ temporal window, $m=(\Delta x,\Delta y,\Delta w,\Delta h)$

* Note that by learning relative movements w.r.t to the spatial anchors at the first frame, we can avoid cumulative errors in conventional tracking algorithms to some extend

## Reference
1. Motion-based
[14] K. Kang, H. Li, J. Yan, X. Zeng, B. Yang, T. Xiao, C. Zhang, Z. Wang, R. Wang, X. Wang, et al. T-cnn: Tubelets with convolutional neural networks for object detection from videos. arXiv preprint arXiv:1604.02532, 2016.
2. Appearance-based
[15] K. Kang, W. Ouyang, H. Li, and X. Wang. Object detection from video tubelets with convolutional neural networks. In CVPR, 2016.
3. seq-nms
[7] W. Han, P. Khorrami, T. L. Paine, P. Ramachandran, M. Babaeizadeh, H. Shi, J. Li, S. Yan, and T. S. Huang. Seq-nms for video object detection. arXiv preprint arXiv:1602.08465, 2016.

## learned
TPN是从RPN衍生而来，它用到了未来的信息，并不能online


# Deep Reinforcement Learning for Visual Object Tracking in Videos
[arXiv](https://arxiv.org/abs/1701.08936)
[TOC]
## Introduction
1. visual tracking requires continuous and accurate predictions in both spatial and temporal domain over a long period of time

## Contributions
1. We propose and develop a novel convolutional recurrent neural network model for visual tracking. The proposed method directly leverages the power of deep-learning models to automatically learn both spatial and temporal constraints.
2. Our framework is trained end-to-end with deep RL algorithms, in which the model is optimized to maximize a tracking performance measure in the long run.
3. Our model is trained fully off-line. When applied to online tracking, only a single forward pass is computed and no online fine-tuning is needed, allowing us to run at frame-rates beyond real-time.
4. Our extensive experiments demonstrate the outstanding performance of our tracking algorithm compared to the state-of-the-art techniques in OTB [29] public tracking benchmark.

## Methods

### Architecture
![DRLT_arch](./.assets/DRLT_arch.jpg)
1. Observation Network: encodes representations of video frames.
   1. $i_t$: The feature vector $i_t$ is typically computed with a sequence of convolutional, pooling, and fully connected layers to encode information about what was seen in this frame.
   2. $s_t$:
       1. When the ground-truth bounding box location is known, such as the first frame in a given sequence, s t is directly set to be the normalized location coordinate (x, y, w, h) ∈ [0, 1], serving as a strong supervising guide for further inferences.
       2. Otherwise, it is padded with zero and only the feature information i t is incorporated by the recurrent network.
2. Recurrent Network: integrates these observations over time and predicts the bounding box location in each frame.

### Training
1. Reward
   1. $r_t = -avg(|l_t-g_t|)-max(|l_t-g_t|)$ (2)
   2. $r_t = \frac{|l_t\cap g_t|}{|l_t\cup g_t|}$ (3)
   >$l_t$ is the location outputted by the recurrent network, $g_t$ is the target ground truth at time $t$,

   3. $R=\sum_{t=1}^Tr_t$
   > (2) is used in the early stage of training, while using the reward definition in (3) in the late stage of training to directly maximize the IoU.

2. Gradient Approximation
   1. parameter $W={W_o,W_r}$
   2. policy function $\pi(l_t|Z_t;W)$
   $Z_t = z_{1:t}=x_1,l_1...x_{t-1},l_{t-1},x_t$ is history of interactions, summarized in the hidden state $h_t$.
   > $x$ is the action

   3. Objective
   $G(W)=E_{p(z_1:T;W)}[\sum_{t=1}^Tr_t]=E_{p(Z_T;W)}[R]$
   4. Gradient
   $\bigtriangledown_WG=\sum_{t=1}^TE_{p(Z_T;W)}[\bigtriangledown_W\ln\pi(l_t|Z_t;W)R]$
   $\bigtriangledown_WG\thickapprox\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\bigtriangledown\ln\pi(l_t^i|Z_t^i;W)R^i$
   $\bigtriangledown_WG\thickapprox\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\bigtriangledown\ln\pi(l_t^i|Z_t^i;W)(R-b_t)
   >$b_t$ is called reinforcement baseline in the RL literature, it is natural to select $b_t = E_π[R_t]$,

   5. Overview
   ![DRLT_algo](./.assets/DRLT_algo.jpg)

## References
### crop-and-regress for tracking
$[3]$ L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, and
P. H. Torr. Fully-convolutional siamese networks for ob-
ject tracking. In European Conference on Computer Vision,
pages 850–865. Springer, 2016.
$[9]$ D. Held, S. Thrun, and S. Savarese. Learning to track at 100
fps with deep regression networks. In European Conference
on Computer Vision, pages 749–765. Springer, 2016.
### Video attention
$[19]$ V. Mnih, N. Heess, A. Graves, et al. Recurrent models of vi-
sual attention. In Advances in Neural Information Processing
Systems, pages 2204–2212, 2014. 1, 3
### Tracking-by-classification methodology
$[14] $Z. Kalal, K. Mikolajczyk, and J. Matas. Tracking-learning-
detection. IEEE transactions on pattern analysis and ma-
chine intelligence, 34(7):1409–1422, 2012.
$[26]$ N. Wang, J. Shi, D.-Y. Yeung, and J. Jia. Understanding and
diagnosing visual tracking systems. In Proceedings of the
IEEE International Conference on Computer Vision, pages
3101–3109, 2015.
### Recurrent-neural-network trackers
$[13]$ S. E. Kahou, V. Michalski, and R. Memisevic. Ratm: Recurrent attentive tracking model. arXiv preprint arXiv:1510.08660, 2015.
$[16]$ M. Kristan, J. Matas, A. Leonardis, M. Felsberg, L. Cehovin, G. Fernandez, T. Vojir, G. Hager, G. Nebehay, and R. Pflugfelder. The visual object tracking vot2015 challenge results. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pages 1–23, 2015.
$[21]$ G. Ning, Z. Zhang, C. Huang, Z. He, X. Ren, and H. Wang. Spatially supervised recurrent convolutional neural networks for visual object tracking. arXiv preprint arXiv:1607.05781, 2016.
### REINFORCE algorithm
$[28]$ R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8(3-4):229–256, 1992.

## Express
1. spatial vicinity (空间上)周边的区域
2. harness (the power of deep-learning models) 驾驭/利用
3. should be fully explored 完全地探究
4. intuition 直觉
5. #, and thus, # 因此
6. formulate &(problem) as # 看做
7. fuse & with # 融合
8. augment & with # 用#来强化/扩大&
9. leverage # to do；leverage # acreoss a new scope (重新)利用
> leverage (vt): to spread or use resources (=money, skills, buildings etc that an organization has available), ideas etc **again** in several different ways or in different parts of a company, system etc

10. benchmark 基准，参考物
11. & (is another useful framework) **apart from** # 除#之外（并列之意）
12. have the potential of doing 有潜力
13. fundamental problem 基本的问题
14. systematic review 系统的回顾
15. & can run at frame-rates beyond real time while maintaining state-of-the-art performance. 实时的同时保持性能
16. two consecutive frames 连续的帧
17. prohibit & from doing 限制/禁止
18. draw inspiration from 启发
19. design # specially tailored for (doing) 为...特制的/专门设计
20. (The importance are) two folds 两方面
21. & is non-trivial task 并不是不重要的任务
22. For simplicity, 简便起见
23. More specifically, 更特别地
24. objective 目标(n.)

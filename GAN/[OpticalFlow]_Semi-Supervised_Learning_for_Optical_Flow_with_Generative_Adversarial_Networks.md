# Semi-Supervised Learning for Optical Flow with Generative Adversarial Networks
[nips](https://papers.nips.cc/paper/6639-semi-supervised-learning-for-optical-flow-with-generative-adversarial-networks.pdf)

<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Semi-Supervised Learning for Optical Flow with Generative Adversarial Networks](#semi-supervised-learning-for-optical-flow-with-generative-adversarial-networks)
  * [Introduction](#introduction)
  * [Semi-Supervised Optical Flow Estimation](#semi-supervised-optical-flow-estimation)
    * [Semi-supervised learning](#semi-supervised-learning)
    * [Adversarial training](#adversarial-training)
  * [References](#references)

<!-- tocstop -->

## Introduction
1. The classical formulation builds upon the assumptions of brightness constancy and spatial smoothness (often do not hold near motion boundaries) [15, 25]
2. our discriminator learns to distinguish the flow warp errors between using the ground truth flow and using the estimated flow

## Semi-Supervised Optical Flow Estimation
### Semi-supervised learning
1. End Point Error (EPE) for the labeled data
$$\mathcal L_{EPE}(f,\hat f)=\sqrt{(u-\hat u)^2+(v-\hat v)^2}$$
2. warping loss and flow smoothness loss
$\mathcal L_{warp}(I_1,I_2,f)=\rho(I_1-\mathbb W(I_2,f))$
$\mathcal L_{smooth}(f)=\rho(\partial_xu)+\rho(\partial_yu)+\rho(\partial_xv)+\rho(\partial_yv)$
> $\partial_x$, $\partial_y$ horizontal and vertical gradient
> $\rho()$ robust penalty function ((e.g., Lorentzian and Charbonnier)
> warping function $\mathbb W(I_2,f)$ uses the bilinear sampling [18] to warp I2 according to the flow field $f$, $I_1-\mathbb W(I_2,f)$ is the flow warp error

3. A baseline semi-supervised learning approach is to minimize $\mathcal L_{EPE}$ for labeled data and minimize $\mathcal L_{warp}$ and $\mathcal L_{smooth}$ for unlabeled data

### Adversarial training
1. The generator $G$ takes a pair of input images to generate optical flow. The discriminator $D$ performs binary classification to distinguish whether a flow warp error image is produced by the estimated flow from the generator G or by the ground truth flow.
![SOF](./.assets/SOF.jpg)

## References
1. brightness constancy and spatial smoothness
[15] B. K. Horn and B. G. Schunck. Determining optical flow. Artificial intelligence, 17(1-3):185–203, 1981.
[25] B. D. Lucas and T. Kanade. An iterative image registration technique with an application to stereo vision. In International Joint Conference on Artificial Intelligence, 1981.
2. real-world videos for training CNNs (unsupervised)
[1] A. Ahmadi and I. Patras. Unsupervised convolutional neural networks for motion estimation. In ICIP, 2016.
[40] J. J. Yu, A. W. Harley, and K. G. Derpanis. Back to basics: Unsupervised learning of optical flow via brightness constancy and motion smoothness. In ECCV Workshops, 2016.
3. Review
[36] D. Sun, S. Roth, and M. J. Black. A quantitative analysis of current practices in optical flow estimation and the principles behind them. IJCV, 106(2):115–137, 2014.
4. GMM to learn the flow warp error at the patch level
[33] D. Rosenbaum and Y. Weiss. Beyond brightness constancy: Learning noise models for optical flow. arXiv, 2016.
5. local statistics
[34] D. Rosenbaum, D. Zoran, and Y. Weiss. Learning the local statistics of optical flow. In NIPS, 2013.
6. FlowNet
[8] P. Fischer, A. Dosovitskiy, E. Ilg, P. Häusser, C. Hazırba¸s, V. Golkov, P. van der Smagt, D. Cremers, and T. Brox. FlowNet: Learning optical flow with convolutional networks. In ICCV, 2015.
[16] E. Ilg, N. Mayer, T. Saikia, M. Keuper, A. Dosovitskiy, and T. Brox. FlowNet 2.0: Evolution of optical flow estimation with deep networks. In CVPR, 2017.
7. SPyNet
[30] A. Ranjan and M. J. Black. Optical flow estimation using a spatial pyramid network. In CVPR, 2017.

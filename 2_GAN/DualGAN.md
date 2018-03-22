# DualGAN: Unsupervised Dual Learning for Image-to-Image Translation


<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](#dualgan-unsupervised-dual-learning-for-image-to-image-translation)
  * [Inspiration](#inspiration)
  * [Objective](#objective)
  * [Target](#target)
* [Training procedure](#training-procedure)
  * [Reference](#reference)
    * [Dual learning](#dual-learning)
    * [Superresolution](#superresolution)
    * [Video prediction](#video-prediction)

<!-- tocstop -->

## Inspiration
dual learning from natural language translation [23]

## Objective


## Target
develop an unsupervised learning framework for general-purpose image-to-image translation, which only relies on unlabeled image data

# Training procedure
1. RMSProp solver
> perform well even on highly nonstationary problems

2. batch: 1~4

## Reference

### Dual learning
[23] Y. Xia, D. He, T. Qin, L. Wang, N. Yu, T.-Y. Liu, and W.-Y. Ma. Dual learning for machine translation. arXiv preprint arXiv:1611.00179, 2016.

### Superresolution
[7] C. Ledig, L. Theis, F. Huszár, J. Caballero, A. Cunningham, A. Acosta, A. Aitken, A. Tejani, J. Totz, Z. Wang, et al. Photo-realistic single image superresolution using a generative adversarial network. arXiv preprint arXiv:1609.04802, 2016.

### Video prediction
[12] M. Mathieu, C. Couprie, and Y. LeCun. Deep multiscale video prediction beyond mean square error. arXiv preprint arXiv:1511.05440, 2015.

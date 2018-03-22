# WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images


<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [WaterGAN: Unsupervised Generative Network to Enable Real-time Color Correction of Monocular Underwater Images](#watergan-unsupervised-generative-network-to-enable-real-time-color-correction-of-monocular-underwater-images)
  * [Abstract](#abstract)
  * [introduction](#introduction)
  * [Approach](#approach)
  * [Reference](#reference)

<!-- tocstop -->

## Abstract
1. Using WaterGAN, we generate a large training dataset of corresponding depth, in-air color images, and realistic underwater images.
## introduction
1. obtaining ground truth of the true color of a natural subsea scene is also an open problem
2. WaterGAN takes in-air images and depth maps as input and generates corresponding synthetic underwater images as output.

## Approach
1. WaterGAN is the first component of this pipeline, taking as input in-air RGB-D images and a sample set of underwater images to train a generative network adversarially
2. architecture
   1. depth estimation network
   the encoder consists of 10 convolution layers and three levels of downsampling. The decoder is symmetric to the encoder, using non-parametric upsampling layers.
   2. color correction network
   To increase the output resolution of our proposed network, we keep the basic network architecture used in the depth estimation stage as the core processing component of our color restoration net. Then we wrap the core component with an extra downsampling and upsampling stage.
   3. For both the depth estimation and color correction networks, a Euclidean loss function is used.
![WG](./.assets/WG.jpg)

## Reference
1. application of underwater vision
[1] M. Johnson-Roberson, M. Bryson, A. Friedman, O. Pizarro, G. Troni, P. Ozog, and J. C. Henderson, “High-resolution underwater robotic vision-based mapping and 3d reconstruction
for archaeology,” J. Field Robotics, pp. 625–643, 2016.
[2] M. Bryson, M. Johnson-Roberson, O. Pizarro, and S. Williams, “Automated registration for multi-year robotic surveys of marine benthic habitats,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots and Syst., 2013, pp. 3344–3349.
2. estimate BL and TM with CNN
[13] Y.-S. Shin, Y. Cho, G. Pandey, and A. Kim, “Estimation of ambient light and transmission map with common convolutional architecture,” in Proc. IEEE/MTS OCEANS, Monterey, CA, 2016, pp. 1–7.
3. SimGAN
[15] A. Shrivastava, T. Pfister, O. Tuzel, J. Susskind, W. Wang, and R. Webb, “Learning from simulated and unsupervised images through adversarial training,” CoRR, vol. abs/1612.07828, 2016.
4. RenderGAN
[16] L. Sixt, B. Wild, and T. Landgraf, “Rendergan: Generating realistic labeled data,” CoRR, vol. abs/1611.01331, 2016.
5. SgNet
[19] V. Badrinarayanan, A. Kendall, and R. Cipolla, “Segnet: A deep convolutional encoder-decoder architecture for scene segmentation,” IEEE Trans. on Pattern Analysis and Machine Intell., vol. PP, no. 99, 2017.
6. Restoration and denosing
[20] X.-J. Mao, C. Shen, and Y.-B. Yang, “Image restoration using convolutional auto-encoders with symmetric skip connections,” ArXiv preprint arXiv:1606.08921, 2016.
[21] V. Jain, J. F. Murray, F. Roth, S. Turaga, V. Zhigulin, K. L. Briggman, M. N. Helmstaedter, W. Denk, and H. S. Seung, “Supervised learning of image restoration with convolutional networks,” in Proc. IEEE Int. Conf. Comp. Vision, IEEE, 2007, pp. 1–8.

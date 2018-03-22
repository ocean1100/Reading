# Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network


<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](#photo-realistic-single-image-super-resolution-using-a-generative-adversarial-network)
  * [Related workd](#related-workd)
  * [Contribution](#contribution)
  * [Perceptual loss function](#perceptual-loss-function)

<!-- tocstop -->

## Related workd
1. Skip-connections relieve the network architecture of modeling the identity mapping that is trivial in nature, however, potentially non-trivial to represent with convolutional kernels.

## Contribution
1. set a new state of the art for image SR with high upscaling factors (4x) as measured by **PSNR** and **structural similarity** (SSIM) with our 16 blocks deep ResNet (SRResNet) optimized for MSE.
2. replace the MSE-based content loss with a loss calculated on feature maps of the VGG network
3. confirm with an extensive mean opinion score (MOS) test on images

## Perceptual loss function
1. VGG loss
$$
l_{VGG/i,j}^{SR}=\frac{1}{W_{i,j}H_{i,j}}\sum_{x=1}^{W_{i,j}}\sum_{y=1}^{H_{i,j}}(\phi_{i,j}(I^{HR}_{x,y})-\phi_{i,j}(G_{\theta_G}(I^{LR}))_{x,y})^2
$$
> $W_{i,j}$ and $H_{i,j}$ describe the dimensions of the respective feature maps within the VGG network.

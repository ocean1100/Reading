# Unsupervised Image-to-Image Translation with Generative Adversarial Networks

## Introduction
1. purposed a two step learning method that utilize conditional GAN to learn the global feature of different domains, and learn to synthesize plausible image from
random noise and given class/domain label
2. proposed a new way of training image encoder.

## Background
1. Discriminator: maximize the probability of assigning the correct label to both training samples and samples generated from generator.
2. Generator: minimize the probability of assigning a samples from generator to be fake, so as to generate plausible samples conditioned on the random noise vector $z$

## Method
1. Training：
![2step](./.assets/2step.png)
## References
1. Y. Taigman, A. Polyak, and L. Wolf. Unsupervised crossdomain image generation. arXiv preprint arXiv:1611.02200, 2016.
> it can also be used for unsupervised domain adaption

2. P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Imageto-image translation with conditional adversarial networks. arXiv preprint arXiv:1611.07004, 2016.
> A common framework

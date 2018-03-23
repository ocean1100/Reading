# Improved Training of Wasserstein GANs
[arXiv](https://arxiv.org/abs/1704.00028)

<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Improved Training of Wasserstein GANs](#improved-training-of-wasserstein-gans)
  * [Contribution](#contribution)
  * [WGAN](#wgan)

<!-- tocstop -->

## Contribution
1. show how critic weight clipping can lead to pathological behavior
2. propose WGAN with gradient penalty
3. demonstrate stable training of many difficult GAN architectures with default settings, performance improvements over weight clipping

## WGAN
1. loss
$$
\min_G\max_{D\in \mathcal D}\mathbb E_{x\backsim \mathbb P_r}[D(x)]-\mathbb E_{\hat x\backsim \mathbb P_g}[D(\hat x)]
$$
> where $\mathcal D$ is the set of 1-Lipschitz functions

2. To enforce the Lipschitz constraint on the critic, WGAN propose to clip the weights of the critic to lie within a compact space $[-c, c]$. The set of functions satisfying this constraint is a subset of the $k$-Lipschitz functions for some $k$ which depends on $c$ and the critic architecture.

[详解](https://www.zhihu.com/question/52602529/answer/158727900)

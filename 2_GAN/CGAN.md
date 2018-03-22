# Conditional Generative Adversarial Nets


<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Conditional Generative Adversarial Nets](#conditional-generative-adversarial-nets)
  * [基本思想](#基本思想)
  * [D的Loss函数对比](#d的loss函数对比)

<!-- tocstop -->

## 基本思想
1. GAN的问题：提出了一种带条件约束的GAN，在生成模型（D）和判别模型（G）的建模中均引入条件变量$y$（conditional variable $y$），使用额外信息y对模型增加条件，可以指导数据生成过程。这些条件变量$y$可以基于多种信息。
2. 如果条件变量$y$是**类别标签**，可以看做CGAN是把纯无监督的 GAN 变成有监督的模型的一种改进。

## D的Loss函数对比
1. GAN
$$ \min_G\max_D V(D,G)=E_{x\in p_{data}}[logD(x)]+E_{x\in p_{z}}[log(1-D(G(z)))]$$
2. CGAN
$$ \min_G\max_D V(D,G)=E_{x\in p_{data}}[logD(x|y)]+E_{x\in p_{z}}[log(1-D(G(z|y)))]$$
> 生成器和判别器都增加额外信息y为条件, y可以使任意信息,例如类别信息,或者其他模态的数据。
![CGAN](./.assets/CGAN.png)

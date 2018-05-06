# Conditional Image-to-Image Translation
[arXiv](https://arxiv.org/abs/1805.00251)

## Problem Setup
1. two domain: $\mathcal D_A, \mathcal D_B$
2. $x_A(\in\mathcal D_A)=x^i_A\oplus x^s_A$
> $x^i_A$: domain-independent features
> $x^s_A$: domain-specific features

3. conditional image-to-image translation:
   1. input: $x_A\in\mathcal D_A$
   2. conditional input: $x_B\in\mathcal D_B$
   3. output: $x_{AB}\in\mathcal D_B$
   >  keeping the domain-independent features of $x_A$ and combining the domain-specific features carried in $x_B$

   $$ x_{AB}=G_{A\to B}(x_A,x_B)= x^i_A\oplus x^s_B$$


## Conditional Dual GAN
![cd_GAN](./.assets/cd_GAN.jpg)

### The Encoder-Decoder Framework
1. encoder $e_A,e_B$: 用于提取特征$x^i,x^s$
2. decoder $g_A,g_B$
$$ x_{AB}=g_B(x_A^i,x_B^s), \quad x_{BA}=g_B(x_B^i,x_A^s) $$

### Training
1. GAN loss
2. Dual learning loss
   1. 重建$x_A,x_B$
   2. reconstruction errors
  $$
  \begin{array}l
  l^{im}_{dual}(x_A,x_B)=||x_A-\hat x_A||^2+||x_B-\hat x_B||^2 \\
  l^{di}_{dual}(x_A,x_B)=||x^i_A-\hat x^i_A||^2+||x^i_B-\hat x^i_B||^2 \\
  l^{ds}_{dual}(x_A,x_B)=||x^s_A-\hat x^s_A||^2+||x^s_B-\hat x^s_B||^2 \\
  \end{array}
   $$

## Learned
1. 和[Fusion GAN](./Generating_a_Fusion_Image_One’s_Identity_and_Another’s_Shape.md)的想法相似
2. 和类似大多数无监督的方法类似, 考虑重建loss和GAN loss

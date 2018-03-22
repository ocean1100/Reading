# Object Detection in Video with Spatiotemporal Sampling Networks
[arXiv](https://arxiv.org/abs/1803.05549)

[TOC]

## Introduction
1. standard convolution
$$
y(p_0) = \sum_{p_n\in R} w(p_n)\cdot x(p_0+p_n)
$$
> $p_n$是 uniformly-spaced grid $R$ 里的元素

2. deformable convolution
$$
y(p_0) = \sum_{p_n\in R} w(p_n)\cdot x(p_0+p_n+\Delta p_n)
$$
> $\Delta p_n$ 是偏移量，也由一个卷积层计算(产生一个单通道 spatial resolution 相同的通道共享的offeset map)

## Spatiotemporal Sampling Network
1. 框架
    1. 静态特征提取
    2. 根据当前帧，在相邻帧(2k个supporting frames，前后各k个)提取相关特征，
    3. 各帧提取出的特征时序地K合(逐像素加权相加)到一个特征张量(作为当前帧的特征)
    4. 这个张量用于检测
![STSN](./.assets/STSN.jpg)
2. Sampling
    1. 前向计算特征$I_{t},I_{t+k}$，产生$f_{t},f_{t+k}$
    2. concatenate $f_{t},f_{t+k}$, 得到$f_{t,t+k}$
    3. 用$f_{t,t+k}$预测location offsets(应该是t和t+k目标间的偏移)，将用于采样$f_{t+k}$
    4. 采样$f_{t+k}$
       1. 输入：预测的location offsets和$f_{t+k}$
       2. 操作：deformable convolution
       3. 输出：sampled feature tensor $g_{t,t+k}$
3. 聚合
   1. 步骤2用于所有2K个supporting frames
   2.
   $$
   \begin{array}{l}
   g_t^{agg} = \sum_{k=-K}^{K}w_{t,t+k}(p)g_{t,t+k}(p) \\
   w_{t,t+k}(p)=\exp(\frac{S(g(t,t))(p)\cdot S(g(t,t+1))(p)}{|S(g(t,t))(p)||S(g(t,t+k))(p)|})
   \end{array}
   $$
   > $S(\cdot)$是一个三层网络，最终,所有$w$计算完后进行softmax操作，s.t. $\sum^K_{k=-K}w_{t,t+k}(p)=1$

4. 训练时$K=1$, 测试时$K=9$

## Reference
1. deformable convolutions
Dai, J., Qi, H., Xiong, Y., Li, Y., Zhang, G., Hu, H., Wei, Y.: Deformable convolutional networks. In: 2017 IEEE International Conference on Computer Vision (ICCV). Volume 00. (Oct. 2018) 764–773

## Learned
也是特征融合，卷积出一个偏移，这个偏移用于deformable convolution，最终得到时序整合的特征。
也是光流预测的代替品, 预测两帧之间的位置变化

# Online Video Object Detection using Association LSTM

## Introduction
1. Association LSTM not only regresses and classifiy directly on object locations and categories but also associates features to represent each output object.
> 不仅仅做回归和分类，还关联特征

## Method
### Architecture
![ALSTM](./.assets/ALSTM.jpg)
1. keep the result of SSD as a $(c+4)$-dimension location-score vector, whose confidence score is above a threshold of 0.8
> $c$是类别数，4是坐标

2. extract a normalized fixedsize descriptor for each detected object using RoI pooling [7]
> 对每个检测框提取一个归一化的定长的描述子。
ROI pooling 输入有两部分组成： data:指的是进入RPN层之前的那个Conv层的Feature Map,通常我们称之为“share_conv”; rois：指的是RPN层的输出，一堆矩形框，形状为1x5x1x1（4个坐标+索引index），其中值得注意的是：坐标的参考系不是针对feature map这张图的，而是针对原图的
输出是batch个vector，其中batch的值等于roi的个数，vector的大小为$channel\times w\times h$；ROI Pooling的过程就是将一个个大小不同的box矩形框，都映射成大小为$w\times h$的矩形框；

3. concatenate the location-score vector and and feature descriptor vector of different objects respectively into a **frame-level tensor** with two dimension $N \times D$, where $D = c+4+s\times s$ is the composite feature length
> 对每个检测框，SSD的输出和ROI描述子拉成一个向量，长度为D。N是检测框的个数，如果不足N个padding 0，如果超过N个只取top N

3. stack the current frame-level tensor with the $\tau-1$ frames backward, yielding a stacked tensor input with size $\tau\times N\times D$
> 联合前$\tau-1$帧的frame-level tensor，产生$\tau\times N\times D$的tensor送给LSTM

4. Each object prediction include three items:
   1. object location in 4-D
   2. category score vector with dimension $c$
   3. and association feature with dimension $s\times s$.

### Objective
1. Object Regression Error
$$
\begin{array}l
\mathcal L_{reg}(l,g,c)=\sum(L_{conf}(c,c^*)+\lambda L_{loc}(l,g))+\alpha \mathcal L_{smooth}  \\
\mathcal L_{smooth} = \sum_\tau(\tilde l_t -\tilde l_{t-1})
\end{array}
$$
> $l$:检测框;$g$:gt框; $L_{loc}$: smmoth $L_1$; $L_{conf}$: softmax loss; $\mathcal L_{smooth}$: 某个时间段内不能变化太大

2.  Association Error
$$
\mathcal L_{asso}=\sum_t \sum_{i,j}\theta_{ji}|\phi^i_{t-1}\cdot \phi^j_{t}|
$$
> $\theta_{ji}\in 0,1$, 仅当$t-1$帧中的第$i$个object与$t$帧中的第$j$个object有关联时(內积最小)$\theta_{ji}=1$。每个$i$只能匹配一个$j$, $\sum_j\theta_{i,j}=1$

### implementation
1. Noting that <span style="color:yellow"> most of the objects are detected using “f c7”, “conv6 2”, “conv7 2” and “conv8 2” feature maps.</span> Thus, we utilize these four feature maps as ROI pooling layers and pool the output boxes onto these feature maps to compute fixed-size feature descriptor
> 找到了对大多数object起作用的layer
2. Data Augmentation
   1. mirroring and random crop with IoU > 0.8
   2. reversed sequence for each trajectory with probability 0.5

## Reference
### LSTM
[8] S. Hochreiter and J. Schmidhuber. Long short-term memory. Neural computation, 9(8):1735–1780, 1997.
### RNN for detection
[3] S. Bell, C. Lawrence Zitnick, K. Bala, and R. Girshick. Inside-outside net: Detecting objects in context with skip pooling and recurrent neural networks. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2016.
<span style="color:yellow"> [26] S. Tripathi, Z. C. Lipton, S. Belongie, and T. Nguyen. Context matters: Refining object detection in video with recurrent neural networks. In Proceedings ofthe British Machine Vision Conference (BMVC), 2016.</span>
[18] G. Ning, Z. Zhang, C. Huang, Z. He, X. Ren, and H. Wang. Spatially supervised recurrent convolutional neural networks for visual object tracking. arXiv preprint arXiv:1607.05781, 2016.
### Batch Normalized LSTM
[5] T. Cooijmans, N. Ballas, C. Laurent, and A. Courville. Recurrent batch normalization. arXiv:1603.09025, 2016.

## Learned
实际上用LSTM做了一个SSD的后处理，把SSD的结果和检测框的特征结合起来送给LSTM，再回归出结果，并可以找到本帧框和前帧框的对应关系

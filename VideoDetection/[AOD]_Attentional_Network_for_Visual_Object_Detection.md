# Attentional Network for Visual Object Detection
[arXiv](https://arxiv.org/abs/1702.01478)

## Method
![AOD](./.assets/AOD.jpg)
1. Architecture
     1. denote a glimpse at each time step $t$ by $G_t$. $G_1$ is a proposal bounding box $(p_x, p_y, p_w, p_h)$.
     $$
     G_t=(\delta_x,\delta_y,\delta_w,\delta_h,)=(\frac{g_x-p_x}{p_w},\frac{g_y-p_y}{p_y},\log\frac{g_w}{p_w},\log\frac{g_h}{p_h},)
     $$
     where $(g_x, g_y, g_w, g_h)$ is the center coordinate, width, height
     2. ROI pooling in $G_t$: The ROI pooling operation divides a given ROI into a predefined grid of sub-windows and then max-pools the feature values in each sub-window.
     3. The pooled features are fed into a stack recurrent neural network of two layers
     4. element-wise max operation
     5. softmax classification layer and bounding box regression layer
2. Reinforcement learning: whether the new glimpse location is useful for the task of object detection or not
$$
r_t=\left\{
\begin{array}l
P(c^*)\times IoU(B_{c^*},B^*_{c^*}) \quad (t=T)  \\
0 \quad (otherwise)
\end{array}
\right.
$$
> $P(c^*)$是正确类别的概率，IoU(intersection over union)是box的重叠程度

$$
R(\xi)=\sum_{t=1}^Tr_t
$$
> $\xi$是一个回合episode，包含$T$ steps ，$R$是回合奖励

$$
J(\pi)=\mathbb E_\pi[R(\xi)]
$$
> $J$是优化木匾函数

$$
\triangledown_\theta J(\pi_\theta)\approx\frac{1}{n}\sum_{i=1}^nR(\xi^{(i)})\sum_{t=1}^n\frac{(a_t^{(i)}-\theta x_t^{(i)})x_t^{(i)T}}{\sigma^2}
$$
> 梯度[22]

## Reference
[22] R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” Machine Learning, vol. 8, no. 3-4, pp. 229–256, 1992.
[25] P. Werbos, “Backpropagation through time: what it does and how to do it,” Proceedings of the IEEE, vol. 78, no. 10, pp. 1550–1560, 1990.
> Back Propagation Through Time (BPTT)

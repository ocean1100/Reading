# Recurrent Neural Network Regularization
[arXiv](https://arxiv.org/abs/1409.2329)

![drop](./.assets/drop.jpg)
1. The main idea is to apply the dropout operator only to the non-recurrent connections
$$
\left(\begin{array}l i\\f\\o\\g\end{array}\right)= \left(\begin{array}l \text{sigmoid}\\\text{sigmoid}\\\text{sigmoid}\\\tanh\end{array}\right)T_{2n,4n}
\left(\begin{array}l \text{Dropout}(h_t^{l-1})\\h^l_{t-1}\end{array}\right)
$$
> 在非循环的部分dropout, $l$ 隐含层，$t$ 时间，对于第一层，应在输入上dropout

2. The optimal dropout probability
   1. MACHINE TRANSLATION: 0.2.

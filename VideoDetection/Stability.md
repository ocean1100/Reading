# On The Stability of Video Detection and Tracking
[arXiv](https://arxiv.org/abs/1611.06467)
[TOC]
## introduction
1. Unfortunately, all these methods are limited in post-processing stage. Few works tried to integrate the temporal context in an end-to-end manner.
> 大部分都是后处理，很少有端到端整合时序信息

## Stability Evaluation Metric
1.
$$
\Phi = E_F+E_C+E_R
$$
2. $E_F$: fragment error
$$
E_F = \frac{1}{N}\sum_{k=1}^N\frac{f_k}{t_k-1}
$$
> the results of a stable detector should be consistent 不能断断续续
> $N$: 总目标数; $t_k$: 第$k$个目标的总长度; $f_K$: 第$k$个目标有几次切换状态（前帧检测到后帧丢失或者反之）

3. $E_C$: center position error
第$f$帧第$k$个目标的预测box: $B_p^{k,f} = (x_p^{k,f},y_p^{k,f},w_p^{k,f},h_p^{k,f})$, gt box:$B_g^{k,f} = (x_g^{k,f},y_g^{k,f},w_g^{k,f},h_g^{k,f})$
$$
\begin{array}l
e_x^{k,f}=\frac{x_p^{k,f}-x_p^{k,f}}{w_g^{k,f}},\sigma^k_x=\text{std}(e^k_x) \\
e_y^{k,f}=\frac{y_p^{k,f}-y_p^{k,f}}{h_g^{k,f}},\sigma^k_x=\text{std}(e^k_y) \\
E_c = \frac{1}{N}\sum_{k=1}^N(\sigma_x^k+\sigma_y^k)
\end{array}
$$
>只考虑了振动情况，未考虑偏移，因为偏移体现在精度中

4. $E_R$: scale and ratio error error
$$
\begin{array}l
e_s^{k,f}=\sqrt{\frac{w_p^{k,f}h_p^{k,f}}{w_g^{k,f}h_g^{k,f}}}, \sigma_s^k=\text{std}(e_s^k) \\
e_r^{k,f}=(\frac{w_p^{k,f}}{h_p^{k,f}})/(\frac{w_g^{k,f}}{h_g^{k,f}}) \\
E_R = \frac{1}{N}\sum_{k=1}^N(\sigma_s^k+\sigma_r^k)
\end{array}
$$

## Referecne
### ConvLSTM for semantic segmentation
[10] M. Fayyaz, M. H. Saffar, M. Sabokrou, M. Fathy, R. Klette, and F. Huang. STFCN: Spatio-temporal fcn for semantic video segmentation. arXiv preprint arXiv:1608.05971, 2016.
[45] S. Valipour, M. Siam, M. Jagersand, and N. Ray. Recurrent fully convolutional networks for video segmentation. arXiv preprint arXiv:1606.00487, 2016.
### Multi-Object Tracking Accuracy (MOTA) and Multi-Object Tracking Precision (MOTP)
[3] K. Bernardin and R. Stiefelhagen. Evaluating Multiple Object Tracking Performance: The CLEAR MOT metrics. EURASIP Journal on Image and Video Processing, 2008(1):1–10, 2008.

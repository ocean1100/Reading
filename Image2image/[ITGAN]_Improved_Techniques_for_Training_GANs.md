# Improved Techniques for Training GANs
[arXiv](https://arxiv.org/abs/1606.03498)
[博客](http://blog.csdn.net/zijin0802034/article/details/58643889)


<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Improved Techniques for Training GANs](#improved-techniques-for-training-gans)
  * [Toward Convergent GAN Training](#toward-convergent-gan-training)

<!-- tocstop -->

## Toward Convergent GAN Training
1. Feature Matching
原始的GAN网络的目标函数需要最大化判别网络的输出。作者提出了新的目标函数，motivation就是让生成网络产生的图片，经过判别网络后的中间层的feature 和真实图片经过判别网络的feature尽可能相同。假定$f(x)$为判别网络中间层输出的feature map。生成网络的目标函数定义如下:
$$ ||\mathbb E_{x\backsim p_{data}} - \mathbb E_{z\backsim p_{z}}f(G(z))||^2_2 $$
判别网络按照原来的方式训练。相比原先的方式，生成网络G产生的数据更符合数据的真实分布。作者虽然不保证能够收敛到纳什均衡点，但是在传统GAN不能稳定收敛的情况下，新的目标函数仍然有效。个人觉得，判别网络从输入到输出逐层卷积，pooling，图片信息逐渐损失，因此中间层能够比输出层得到更好的原始图片的分布信息，拿中间层的feature作为目标函数比输出层的结果，能够生成图片信息更多。可能采用这种目标函数，生成的图片会效果会更好。

2. MiniBatch discrimination
判别网络如果每次只看单张图片，如果判断为真的话，那么生成网络就会认为这里一个优化的目标，导致生成网络会快速收敛到当前点。作者使用了minibatch的方法，每次判别网络输入一批数据进行判断。假设$f(x)\in R^{A}$表示判别网络中间层的输出向量。作者将$f(x)$乘以矩阵$T\in R^{A\times B\times C}$，得到一个矩阵$M_i\in R^{B\times C}$。计算矩阵$M_i$每行的$L_1$距离， 到$c_b(x_i,x_j)=exp(-||M_{ib}-M_{jb}||_{L1} \in R$ 。 定义输入$x_i$的输出$o(x_i)$如下：
$$
\begin{array}r
o(x_i)_b = \sum_{j=1}^nc_b(x_i, x_j)\in R \\
o(x_i)=[o(x_i)_1, ..., o(x_i)_n] \\
o(X)\in R^{n\times B}
\end{array}
$$
将$o(x_i)$作为输入，进入判别网络下一层的输入。

3. Historical averaging
在生成网络和判别网络的损失函数中添加一个项：
$$ ||\theta - \frac{1}{t}\sum_{i=1}^{t}\theta[i]||^2 $$
公式中$\theta[i]$表示在$i$时刻的参数。这个项在网络训练过程中，也会更新。加入这个项后，梯度就不容易进入稳定的轨道，能够继续向均衡点更新。

4. One-side label smooth
将正例label乘以$\alpha$,， 负例label乘以$\beta$，最优的判别函数分类器变为:
$$ D(x) = \frac{\alpha p_{data}(x)+\beta p_{model}(x)}{p_{data}(x)+p_{model}(x)} $$

5. Virtual batch normalization
BN使用能够提高网络的收敛，但是BN带来了一个问题，就是layer的输出和本次batch内的其他输入相关。为了避免这个问题，作者提出了一种新的bn方法，叫做virtual batch normalization。首先从训练集中拿出一个batch在训练开始前固定起来，算出这个特定batch的均值和方差，进行更新训练中的其他batch。VBN的缺点也显而易见，就是需要更新两份参数，比较耗时。

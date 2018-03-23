
# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
[arXiv](https://arxiv.org/abs/1703.10593)

<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Objective](#objective)
* [Implementation](#implementation)
  * [Network Architecture](#network-architecture)
* [References](#references)
  * [Supervised image-to-image](#supervised-image-to-image)
  * [Cycle consistency loss](#cycle-consistency-loss)
  * [The beginning of image-to-image](#the-beginning-of-image-to-image)
  * [Neural Style Transfer](#neural-style-transfer)
  * [Adam](#adam)
  * [Baselines](#baselines)
* [Code](#code)

<!-- tocstop -->


## Objective
1. Cycle Consistency Loss
![cycle](./.assets/cycle.png)
$$ \mathcal L_{cyc}(G,F)=\mathbb E_{x\backsim p_{data}(x)}[||F(G(x))-x||_1]+\mathbb E_{y\backsim p_{data}(y)}[||F(G(y))-y||_1]$$
> There is no improved performance, if replace the L1 norm in this loss with an adversarial loss between $F(G(x))$ and $x$, and between $G(F(y))$ and $y$

2. Full Objective
$$
\begin{array}l
\mathcal L(G,F,D_X,D_Y)=\mathcal L_{GAN}(G,D_Y,X,Y)+\mathcal L_{GAN}(F,D_X,X,Y)+\lambda\mathcal L_{cyc}(G,F) \\
G^*,F^*=arg\min_{G,F}\max_{D_X,D_Y}\mathcal L(G,F,D_X,D_Y)
\end{array}
$$
> $ \lambda= 10$

3. identity loss
$$ \mathcal L_{identity}(G,F)=\mathbb E_{y\backsim p_{data}(y)}[||G(y)-y||_1]+\mathbb E_{x\backsim p_{data}(x)}[||F(x)-x||_1] $$
> preserve color composition between the input and output. Without $\mathcal L_{identity}$, the generator G and F are free to change the tint of input images when there is no need to.

## Implementation
### Network Architecture
1. Johnson et al. [22]
2. We use 6 blocks for 128*128 images, and 9 blocks for 256*256 and higherresolution training images.
3. For the discriminator networks we use 70*70 PatchGANs, which aim to classify whether 70*70 overlapping image patches are real or fake. patch-level discriminator architecture has fewer parameters than a full-image discriminator, and can be applied to arbitrarily-sized images in a fully convolutional fashion
4. Training details
   1. replace the negative log likelihood objective by a least square loss
    $$\mathcal L_{LSGAN}(G,D_Y,X,Y)=\mathbb E_{y\backsim p_{data}(y)}[(D_Y(y)-1)^2]+\mathbb E_{x\backsim p_{data}(x)}[D_Y(G(x))^2]$$
    2. To reduce model oscillation: update the discriminators $D_X$ and $D_Y$ using a history of generated images rather than the ones produced by the latest generative networks.

## References

### Supervised image-to-image
[22] J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, pages 694–711. Springer, 2016.
[55] X. Wang and A. Gupta. Generative image modeling using style and structure adversarial networks. ECCV, 2016.
[61] R. Zhang, P. Isola, and A. A. Efros. Colorful image colorization. In ECCV, 2016.

### Cycle consistency loss
[65] J.-Y. Zhu, P. Kr¨ahenb¨uhl, E. Shechtman, and A. A. Efros. Generative visual manipulation on the natural image manifold. In ECCV, 2016.

### The beginning of image-to-image
[18] A. Hertzmann, C. E. Jacobs, N. Oliver, B. Curless, and D. H. Salesin. Image analogies. In SIGGRAPH, pages 327–340. ACM, 2001.
> a nonparametric texture model

[31] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, pages 3431–3440, 2015.
> a parametric translation function using CNNs

### Neural Style Transfer
[11] L. A. Gatys, M. Bethge, A. Hertzmann, and E. Shechtman. Preserving color in neural artistic style transfer. arXiv preprint arXiv:1606.05897, 2016.
[12] L. A. Gatys, A. S. Ecker, and M. Bethge. Image style transfer using convolutional neural networks. CVPR, 2016.
**[22]** J. Johnson, A. Alahi, and L. Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, pages 694–711. Springer, 2016.
[51] D. Ulyanov, V. Lebedev, A. Vedaldi, and V. Lempitsky. Texture networks: Feed-forward synthesis of textures and stylized images. In Int. Conf. on Machine
Learning (ICML), 2016.

### Adam
[24] D. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

### Baselines
1. CoGAN
[30] M.-Y. Liu and O. Tuzel. Coupled generative adversarial networks. In NIPS, pages 469–477, 2016.
>learns a joint distribution over images from two domains.

2. SimGAN, Feature loss + GAN
[45] A. Shrivastava, T. Pfister, O. Tuzel, J. Susskind, W. Wang, and R. Webb. Learning from simulated and unsupervised images through adversarial training. arXiv preprint arXiv:1612.07828, 2016.
3. BiGAN/ALI
[8] V. Dumoulin, I. Belghazi, B. Poole, A. Lamb, M. Arjovsky, O. Mastropietro, and A. Courville. Adversarially learned inference. arXiv preprint arXiv:1606.00704, 2016.
4. ALI
[6] J. Donahue, P. Kr¨ahenb¨uhl, and T. Darrell. Adversarial feature learning. arXiv preprint arXiv:1605.09782, 2016.

## Code
1. image pool
```python
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0) # Returns a new tensor with a dimension of size one inserted at the specified position.
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: # 在pool中任选batch_size个
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image) # 返回当前的
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
```

2. loss
```python
def backward_D_A(self):
    fake_B = self.fake_B_pool.query(self.fake_B)
    #self.fake_B, fake_B 都是 torch.cuda.FloatTensor of size batch_sizex3x256x256
    self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
```

# Image-to-Image Translation with Conditional Adversarial Networks
[arXiv](https://arxiv.org/abs/1611.07004)

<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Image-to-Image Translation with Conditional Adversarial Networks](#image-to-image-translation-with-conditional-adversarial-networks)
  * [Introduction](#introduction)
  * [Related work](#related-work)
  * [Method](#method)
  * [Experiment](#experiment)
  * [References](#references)
  * [Pytorch 源码](#pytorch-源码)

<!-- tocstop -->

## Introduction
1. we must be careful what we wish for (tell the CNN what we wish it to minimize)! If we take a naive approach, and ask the CNN to minimize Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results [29, 46]. This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring.
> 需要谨慎对待CNN的Loss函数

2. Contribution
   1. GANs can be as a **general-purpose solution** for image-toimage translation.
   2. present a simple framework sufficient to achieve good results, and to analyze the effects of several important architectural choices.
> GAN在image-to-image translation problems上的通用性

## Related work
1. Conditional GANs instead learn a structured loss (penalize the joint configuration of the output).
2. we use a “U-Net”-based architecture, and we investigate the effect of changing the patch size

## Method
1. conditional GANs learn a mapping from observed image $x$ and random noise vector $z$, to output image $y$: $G :\{x; z\} \to y$.
> the input z to the generator is sampled from some simple noise distribution, such as the uniform distribution or a **spherical Gaussian distribution**，后者更好

2. Objective
$$
\begin{array}{rl}
\mathcal L_{CGAN} &= E_{x,y\backsim p_{data}(x,y)}[log D(x,y)]+E_{x\backsim p_{data(x)},z\backsim p_z(z)}[log(1-D(x,G(x,z)))] \\
G^*&=arg\min_G\max_D \mathcal L_{CGAN}(G,D)
\end{array}
$$
3. Objective with L1 distance
$$
\begin{array}{rl}
\mathcal{L}_{L1}(G)&= E_{x\backsim p_{data(x)},z\backsim p_z(z)}[||y-G(x,z)||_1] \\
G^*&=arg\min_G\max_D \mathcal L_{CGAN}(G,D) +\lambda\mathcal{L}_{L1}(G)
\end{array}
$$
> 加入了$L_{1}$约束项，使生成图像不仅要像真实图片，也要更接近于输入的条件图片。

3. Network architectures
Let $C_k$ denote a Convolution-BatchNorm-ReLU layer with k filters. $CD_k$ denotes a a Convolution-BatchNorm-Dropout-ReLU layer with a dropout rate of $50\%$. All convolutions are $4\times4$ spatial filters applied with stride $2$. Convolutions in the encoder, and in the discriminator, downsample by a factor of $2$, whereas in the decoder they upsample by a factor of $2$.
   1. Generator
    encoder:
    C64-C128-C256-C512-C512-C512-C512-C512
    decoder:
    CD512-CD512-CD512-C512-C512-C256-C128-C64
    U-Net decoder:
    CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    >在Image-to-Image Translation的大多任务中，图像的底层特征同样重要，所以利用U-net代替encoder-decoder。
    ![UNet2](./.assets/UNet2.png)
    ![UNet](./.assets/UNet.jpg)

    2. Discriminator
    $70\times 70$: C64-C128-C256-C512
    $1\times 1$: C64-C128
    $16\times 16$: C64-C128
    $256\times 256$: C64-C128-C256-C512-C512-C512
4. PatchGAN
> effective in capturing local high-frequency features such as texture and
style, but less so in modeling global distributions.

   1. it is sufficient to restrict our attention to the structure in local image patches.
   2. a discriminator architecture – which we term a PatchGAN – that only penalizes structure at the scale of patches.
   3. run this discriminator convolutationally across the image, averaging all responses to provide the ultimate output of D.
   4. a smaller PatchGAN has fewer parameters, runs faster, and can be applied on arbitrarily large images.
> 通常判断都是对生成样本整体进行判断，比如对一张图片来说，就是直接看整张照片是否真实。而且Image-to-Image Translation中很多评价是像素对像素的，所以在这里提出了分块判断的算法，在图像的每个$N\times N$块上去判断是否为真，最终平均给出结果。

## Experiment
1. Evaluation metrics
   1. AMT perceptual studies
   2. FCN-score
   > FCN-scores 越高，表示图片中有更多可辨认的物体。

2. Analysis of the objective function
   1. A U-Net architecture allows low-level information to shortcut across the network.
3. PixelGANs, PatchGANs, ImageGANs
   1. Color histogram matching is a common problem in image processing

## References
1. conditional random fields:
  L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille. Semantic image segmentation with deep convolutional nets and fully connected crfs
2. Application in image
   1. image prediction from a normal map
   [39] X. Wang and A. Gupta. Generative image modeling using style and structure adversarial networks
   2. image manipulation guided by user constraints
   [29] J.-Y. Zhu, P. Kr¨ahenb¨uhl, E. Shechtman, and A. A. Efros. Generative visual manipulation on the natural image manifold.
   3. future frame prediction
   [27] M. Mathieu, C. Couprie, and Y. LeCun. Deep multi-scale video prediction beyond mean square error.
   4. future state prediction
   [48] Y. Zhou and T. L. Berg. Learning temporal transformations from time-lapse videos.
   5. product photo generation
   [43] D. Yoo, N. Kim, S. Park, A. S. Paek, and I. S. Kweon. Pixellevel domain transfer. ECCV, 2016.
   6. style transfer
   [25] C. Li and M. Wand. Precomputed real-time texture synthesis with markovian generative adversarial networks. ECCV, 2016.
   > Patch-GAN

   7. inpainting
   [29] D. Pathak, P. Krahenbuhl, J. Donahue, T. Darrell, and A. A. Efros. Context encoders: Feature learning by inpainting. CVPR, 2016.
3. “U-Net”
O. Ronneberger, P. Fischer, and T. Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI, pages 234¨C241. Springer, 2015.

## Pytorch 源码
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
1. Network
   1. D
   ```python
   def define_D(c, ndf, which_model_netD,
                n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
   # input_nc = opt.input_nc + opt.output_nc 输入通道数
   # ndf = 64 discrim filters in first conv layer
   # which_model_netD = basic or n_layers
   # n_layers_D only used if which_model_netD==n_layers
       netD = None
       use_gpu = len(gpu_ids) > 0
       norm_layer = get_norm_layer(norm_type=norm)

       if use_gpu:
           assert(torch.cuda.is_available())
       if which_model_netD == 'basic':
           netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
       elif which_model_netD == 'n_layers':
           netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
       else:
           raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                     which_model_netD)
       if use_gpu:
           netD.cuda(device_id=gpu_ids[0])
       init_weights(netD, init_type=init_type)
       return netD
     # Defines the PatchGAN discriminator with the specified arguments.
   class NLayerDiscriminator(nn.Module):
       def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
           super(NLayerDiscriminator, self).__init__()
           self.gpu_ids = gpu_ids
           if type(norm_layer) == functools.partial:
               use_bias = norm_layer.func == nn.InstanceNorm2d
           else:
               use_bias = norm_layer == nn.InstanceNorm2d
           kw = 4
           padw = 1
           sequence = [
               nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
               nn.LeakyReLU(0.2, True)
           ]
           nf_mult = 1
           nf_mult_prev = 1
           for n in range(1, n_layers):
               nf_mult_prev = nf_mult
               nf_mult = min(2**n, 8)
               sequence += [
                   nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                             kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                   norm_layer(ndf * nf_mult),
                   nn.LeakyReLU(0.2, True)
               ]
           nf_mult_prev = nf_mult
           nf_mult = min(2**n_layers, 8)
           sequence += [
               nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=kw, stride=1, padding=padw, bias=use_bias),
               norm_layer(ndf * nf_mult),
               nn.LeakyReLU(0.2, True)
           ]
           sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
           if use_sigmoid:
               sequence += [nn.Sigmoid()]

           self.model = nn.Sequential(*sequence)
   ```
  >NLayerDiscriminator (
  (model): Sequential (
    (0): Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU (0.2, inplace)
    (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
    (4): LeakyReLU (0.2, inplace)
    (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (6): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    (7): LeakyReLU (0.2, inplace)
    (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
    (9): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    (10): LeakyReLU (0.2, inplace)
    (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
    (12): Sigmoid ()
  )
  )
Total number of parameters: 2768705 最后一层输出为$30*30$，感受阈为$70*70$

   2. G
   ```python
   class UnetGenerator(nn.Module):
       def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                    norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
  # unet256参数：3, 3, 8, 64, batch, True
           super(UnetGenerator, self).__init__()
           self.gpu_ids = gpu_ids

           # construct unet structure
           unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
           # unet256参数：512， 512， none, none, batch, True
           for i in range(num_downs - 5):
               unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
           unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
           unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
           unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
           unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

           self.model = unet_block
    # Defines the submodule with skip connection.
    # X -------------------identity---------------------- X
    #   |-- downsampling -- |submodule| -- upsampling --|
    class UnetSkipConnectionBlock(nn.Module):
        def __init__(self, outer_nc, inner_nc, input_nc=None,
                     submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
            super(UnetSkipConnectionBlock, self).__init__()
            self.outermost = outermost
            if type(norm_layer) == functools.partial:
                use_bias = norm_layer.func == nn.InstanceNorm2d
            else:
                use_bias = norm_layer == nn.InstanceNorm2d
            if input_nc is None:
                input_nc = outer_nc
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1, bias=use_bias)
            downrelu = nn.LeakyReLU(0.2, True)
            downnorm = norm_layer(inner_nc)
            uprelu = nn.ReLU(True)
            upnorm = norm_layer(outer_nc)

            if outermost:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]

                if use_dropout:
                    model = down + [submodule] + up + [nn.Dropout(0.5)]
                else:
                    model = down + [submodule] + up

            self.model = nn.Sequential(*model)

        def forward(self, x):
            if self.outermost:
                return self.model(x)
            else:
                return torch.cat([x, self.model(x)], 1)
   ```

3. 单步调试G unet256
   1. `unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)`
    downconv = Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    downrelu = LeakyReLU (0.2, inplace)
    downnorm = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    uprelu = ReLU (inplace)
    upnorm = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    upconv = ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    down = [downrelu, downconv]
    up = [uprelu, upconv, upnorm]
    model = down + up
    > Sequential (
    (0): LeakyReLU (0.2, inplace)
    (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (2): ReLU (inplace)
    (3): ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
    (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
  )

   2. unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
   downconv = Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
   downrelu = LeakyReLU (0.2, inplace)
   downnorm = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
   uprelu = ReLU (inplace)
   upnorm = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
   upconv = ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
   down = [downrelu, downconv, downnorm]
   up = [uprelu, upconv, upnorm]
   model = down + [submodule] + up + [torch.nn.Dropout(0.5)]
   > Sequential (
  (0): LeakyReLU (0.2, inplace)
  (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  (2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
  (3): UnetSkipConnectionBlock (
    (model): Sequential (
      (0): LeakyReLU (0.2, inplace)
      (1): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (2): ReLU (inplace)
      (3): ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
    )
  )
  (4): ReLU (inplace)
  (5): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
  (6): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
  (7): Dropout (p = 0.5)
)

   3.unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
   downconv = Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
   downrelu = LeakyReLU (0.2, inplace)
   downnorm = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
   uprelu = ReLU (inplace)
   upnorm = BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True)
   upconv = ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
   down = [downconv]
   up = [uprelu, upconv, torch.nn.Tanh()]
   model = down + [submodule] + up
   > Sequential (
     (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
     (1): UnetSkipConnectionBlock
     ...
     (2): ReLU (inplace)
     (3): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
     (4): Tanh ()


2. Loss
   1. G_GAN
   > G(A) should fake the discriminator

    ```python
    fake_AB = torch.cat((self.real_A, self.fake_B), 1) # 在channel的维度上cat;  torch.cuda.FloatTensor of size barch_size x channel*2 x 256 x 256
    pred_fake = self.netD.forward(fake_AB) #torch.cuda.FloatTensor of size barch_sizex 1 x 30x 30
    self.loss_G_GAN = self.criterionGAN(pred_fake, True)

    # anout criterionGAN
    self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

    # anout criterionGAN 是一个可调用的类
    class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            # numel()返回数据点的个数=batch_size*1*30*30
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                # 全填1
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor
      # torch.cuda.FloatTensor of size batch_sisex1x30x30 每个30x30全0或全1

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

    # about BCELoss:
    # loss(o, t) = - 1/n \sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

    ```
    2. G_L1
    > Second, G(A) = B

    ```python
    self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) *self.opt.lambda_A

    # about L1Loss
    # loss(x, y) = 1/n \sum |x_i - y_i|

    # combine
    self.loss_G = self.loss_G_GAN + self.loss_G_L1
    self.loss_G.backward()
    ```
    3. D
    ```python
    # Fake
    # stop backprop to the generator by detaching fake_B
    fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
    self.pred_fake = self.netD.forward(fake_AB.detach())
    self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

    # Real
    real_AB = torch.cat((self.real_A, self.real_B), 1)
    self.pred_real = self.netD.forward(real_AB)
    self.loss_D_real = self.criterionGAN(self.pred_real, True)

    # Combined loss
    self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    self.loss_D.backward()
    ```
3. Learning rate
```python
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda': # 自定义下降函数
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay+1)
            return lr_l
        # lr = lr*lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        # lr = lr * gamma^(epoch/step)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
```

4. ResnetGenerator
   1. ReflectionPad2d: Pads the input tensor using the reflection of the input boundary
   - Input: :math:`(N, C, H_{in}, W_{in})`
   - Output::math:`(N, C, H_{out}, W_{out})` where
            :math:`H_{out} = H_{in} + paddingTop + paddingBottom`
            :math:`W_{out} = W_{in} + paddingLeft + paddingRight`
    2. Conv2d
    - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
    - Output::math:`(N, C_{out}, H_{out}, W_{out})` where
             :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
             :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
    3. ConvTranspose1d
    - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
    - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
              :math:`H_{out} = (H_{in} - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]`
              :math:`W_{out} = (W_{in} - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]`

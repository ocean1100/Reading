# YOLOv3: An Incremental Improvement
[paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
[pytorch](https://blog.paperspace.com/tag/series-yolo/)

## Deal
1. 不使用softmax，开放式分类
2. 用3个尺度特征来预测，每个尺度的每个像素点预测3个box，
   $ N\times N \times [3*(4+1+80)] $
   > 4: bbox offset, 1: 是否为object， 80: class

3. 使用高层特征和底层特征融合
take the feature map from 2 layers previous and upsample it by 2×. We also take a feature map from earlier in the network and merge it with our upsampled features using element-wise addition.
>目的: meaningful semantic information + finer-grained information

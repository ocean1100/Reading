# Faster RCNN


<!-- toc orderedList:0 depthFrom:1 depthTo:6 -->

* [Faster RCNN](#faster-rcnn)
  * [Others’ Note](#others-note)
  * [运行py-faster-rcnn](#运行py-faster-rcnn)

<!-- tocstop -->


## Others’ Note
1. 介绍
   1. Faster-RCNN由RPN和Fast-RCNN组成，RPN负责寻找proposal，Fast-RCNN负责对RPN的结果进一步优化
2. RPN(一个全卷积网络FCN)
   1. anchor: 把这个feature map上的每一个点映射回原图，得到这些点的坐标，然后着这些点周围取一些提前设定好的区域，如选取每个点周围5x5的一个区域，这些选好的区域可以用来训练RPN。假设我们对feature map上的每个点选取了K个anchor，feature map的大小为H*W*C，那么我们再对这个feature map做两次卷积操作，输出分别是H*W*num_class*K和H*W*4*K，分别对应每个点每个anchor属于每一类的概率以及它所对应的物体的坐标。

## 运行py-faster-rcnn
1. 修改ROOT/caffe-fast-rcnn/make.config，解注释
```
USE_CUDNN := 1
WITH_PYTHON_LAYER := 1
```
2. git clone <caffe-mater>
3.  
```
cd ~/Documents/py-faster-rcnn/caffe-fast-rcnn
mkdir ./include/caffe/layers
cp ~/Documents/caffe/include/caffe/layers/* ./include/caffe/layers
cp ~/Documents/caffe/src/caffe/layers/cudnn_* ./src/caffe/layers
cp ~/Documents/caffe/include/caffe/util/cudnn.hpp ./include/caffe/util
make all -j8
```

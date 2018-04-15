# SteadyFlow: Spatially Smooth Optical Flow for Video Stabilization
[author](http://www.liushuaicheng.org/)

## Introduction
1. Goal: smooth pixel profiles instead of smoothing feature trajectories

## SteadyFlow
1. Goal：
   1. be close to the raw optical flow
   2. be spatially smooth to avoid distortions
2. Estimation
   1. initialize the SteadyFlow by a robust optical flow estimation
   2. identify discontinuous motion vectors and overwrite them by interpolating the motion vectors from neighboring pixels.
      1. we threshold the gradient magnitude of raw optical flow to identify discontinuous regions
      > 用梯度的大小来确定discontinuous regions，>0.1的像素为discontinuous

      2. in a stable video, the accumulated motion vectors $c_t(p)=\sum_tu_t(p)$ should be smooth over time
      > t是时间、p是像素点、u是motion vector，如果$c_t(p)$振荡过大，p看做discontinuous

   3. pixel profiles based stabilization is applied on the SteadyFlow

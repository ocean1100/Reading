# Object Detection in Videos by Short and Long Range Object Linking
[arXiv](https://arxiv.org/abs/1801.09823)

## Approach
1. Cuboid proposal for a short segment
   propose 一个立方体
2. Tubelet detection for a short segment
   1. 对这个立方体回归、分类，得分是所有bbox得分的聚合
   2. 聚合方式：$0.5(mean+max)$
3. Tubelet non-maximum suppression
   1. 2D NMS 易导致tubelet中断
   2. tubelets 的IOU：先计算对应帧的bbox的IOU，取其中最小的
4. Classification refinement through temporally-overlapping tubelets.
   1. M个 short segments(包含1~K帧，K~2K-1帧 ...), 都有Cuboid proposal
   2. K帧bbox重叠较大的两个bbox所在的tubelets看做是同一个
   3. 按照2的方法融合两段tubelets的得分

## Learned
提出了一些规则，人为地考虑了时序一致性

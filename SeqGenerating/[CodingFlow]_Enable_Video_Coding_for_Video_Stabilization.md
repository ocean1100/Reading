# CodingFlow: Enable Video Coding for Video Stabilization
[ieee](https://ieeexplore.ieee.org/document/7909025/?arnumber=7909025&source=authoralert)
[author](http://www.liushuaicheng.org/)

## Introduction
1. The existing video stabilization methods heavily rely on image features [10] for the recovery of camera motions (**time-consuming**)
    1. match image features between neighboring frames [1,5]
    2. track them for a certain range of frames [2,3]
    3. gyroscopes [12,13]
    4. single parametric transformation model [1,7,16]
    5. smoothing long feature tracks [2,3,11,17]
    6. smoothing multiple transformation models [5,6,8] (multiple affines [8], homographies [5], or even nonparametric dense flows [6] )
2. 3D:
    1. 3D reconstruction [2]
    2. a depth camera and light-field camera [30,31]
    3. plane constraints [9,32]
    4. gyroscopes for 3D orientations [12,13]
    5. constrained the 3D rotations [33]
3. 2.5D
    1. smooth feature tracks in a subspace [3]
    2. epipolar geometry for 3D coherence [11]

3. 2D [1], [5]–[7], [16]
    1. estimated a single homography between two adjacent frames [7]
    2. rolling shutter correction [8]
    3. spatially-variant motion estimation [5]
    4. optical flow [6]
    5. sparse flow according to image feature matches [35]


## Reference
[1] Y. Matsushita, E. Ofek, W. Ge, X. Tang, and H.-Y. Shum, “Full-frame video stabilization with motion inpainting,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 28, no. 7, pp. 1150–1163, Jul. 2006.

[2] F. Liu, M. Gleicher, H. Jin, and A. Agarwala, “Content-preserving warps for 3D video stabilization,” ACM Trans. Graph., vol. 28, no. 3, 2009, Art. no. 44.

[3] F. Liu, M. Gleicher, J. Wang, H. Jin, and A. Agarwala, “Subspace video stabilization,” ACM Trans. Graph., vol. 30, no. 4, 2011, Art. no. 4.

[4] B.-Y. Chen, K.-Y. Lee, W.-T. Huan, and J.-S. Lin, “Capturing intentionbased full-frame video stabilization,” Comput. Graph. Forum, vol. 27, no. 7, pp. 1805–1814, Oct. 2008.

[5] S. Liu, L. Yuan, P. Tan, and J. Sun, “Bundled camera paths for video stabilization,” ACM Trans. Graph., vol. 32, no. 4, 2013, Art. no. 78.

[6] S. Liu, L. Yuan, P. Tan, and J. Sun, “Steadyflow: Spatially smooth optical flow for video stabilization,” in Proc. CVPR, 2014, pp. 4209–4216.

[7] M. Grundmann, V. Kwatra, and I. Essa, “Auto-directed video stabilization with robust L1 optimal camera paths,” in Proc. CVPR, 2011, pp. 225–232.

[8] M. Grundmann, V. Kwatra, D. Castro, and I. Essa, “Calibration-free rolling shutter removal,” in Proc. ICCP, 2012, pp. 1–8.

[9] Z. Zhou, H. Jin, and Y. Ma, “Plane-based content-preserving warps for video stabilization,” in Proc. CVPR, 2013, pp. 2299–2306.

[12] S. Bell, A. Troccoli, and K. Pulli, “A non-linear filter for gyroscopebased video stabilization,” in Proc. ECCV, 2014, pp. 294–308.

[13] A. Karpenko, D. E. Jacobs, J. Baek, and M. Levoy, “Digital video stabilization and rolling shutter correction using gyroscopes,” Stanford Comput. Sci., Stanford, CA, USA, Tech. Rep. CSTR 2011-03, 2011.

## Learnd
本文结合了coding和flow两个领域

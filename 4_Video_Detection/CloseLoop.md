# Spatio-Temporal Closed-Loop Object Detection

## Contribution
1. closed-loop proposals: exploit not only the current frame visual feature but also the proposals evaluated on a previous frame
2. improve both detection quality and speed
> 提出了closed-loop proposals，是一种区域提出方法，考虑当前帧和前帧的信息

## Methods
![fedback](./.assets/fedback.jpg)

## Reference
### video objectness proposal
[29] D. Oneata, J. Revaud, J. Verbeek, and C. Schmid, “Spatio-temporal object detection proposals,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 737–752.
> region growing method

[40] M. Van den Bergh, G. Roig, X. Boix, S. Manen, and L. Van Gool, “Online video SEEDS for temporal window objectness,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Dec. 2013, pp. 377–384.
> computationally expensive and requires to process the whole video
tracking windows, online optimization,  30fps

### comparison of twelve object proposal methods for images
[21] J. Hosang, R. Benenson, P. Dollár, and B. Schiele. (2015). “What makes for effective detection proposals?” [Online]. Available: https://arxiv.org/abs/1502.05082

### Video localization
[23] A. Joulin, K. Tang, and L. Fei-Fei, “Efficient image and video colocalization with Frank-Wolfe algorithm,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2014, pp. 253–268.

### detection then tracking
[25] S. Kwak, M. Cho, I. Laptev, J. Ponce, and C. Schmid, “Unsupervised object discovery and tracking in video collections,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2015, pp. 3173–3181.
> object discovery and tracking

### Video detection
[37] S. Tripathi, S. Belongie, Y. Hwang, and T. Nguyen, “Detecting temporally consistent objects in videos through object class label propagation,” in Proc. IEEE Winter Conf. Appl. Comput. Vis. (WACV), Mar. 2016, pp. 1–9.
> extends EdgeBoxes from image object proposals to videos exploiting temporal edge responses

## Learned
直接把前帧的检测结果送给RPN产生新的结果

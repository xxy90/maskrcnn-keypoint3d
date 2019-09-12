## Description
This is pytorch implemtation for maskrcnn predict keypoints and corresponding depths

## For Training
run python main.py


## predict 

run python kp3d_predict.py




##the modification of maskrcnn

1. the model function is keypoints_rcnn_3d.py ,in which we add a depth branch

2. mainly focus on the roi_heads.py ,in which we add depths loss ,similar to the keypoints loss


##the another way to add depth

3. the model function is keypoints_rcnn_3d2.py ,in which we add a depth branch,but share the roi_pool and head with keypoints

4. the roi_heads2.py ,in which we add a depth loss ,but share the keypoint_feature  

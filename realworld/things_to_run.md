hk-19nli010000585158b81c8e794b2fddea0ec478d75afc0a1
## realsense
```
cindy@cindy-IdeaPad-Linux:~/Documents/R2S2R/realsense-ros$ ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true rgb_camera.color_profile:=1280x720x15 depth_module.depth_profile:=1280x720x15 pointcloud.enable:=true
```

## SoM
```
export all_proxy=''
export ALL_PROXY=''
(SAM2) cindy@cindy-IdeaPad-Linux:~/Documents/R2S2R/Enhanced_ReKep4xarm_Tinker/SoM$ python som_gpt4v/main_som.py grasp
```
对准之后直接按回车，等待结果，然后再跑rekep main

## AnyGrasp
```
(AnyGrasp) cindy@cindy-IdeaPad-Linux:~/Documents/R2S2R/anygrasp_sdk/grasp_detection$ python detector_ros.py 
```

## TAPNet
```
(TAPNet) cindy@cindy-IdeaPad-Linux:~/Documents/R2S2R/Enhanced_ReKep4xarm_Tinker/point_tracker$ python point_track_ros.py 
```


## ReKep
```
export all_proxy=''
export ALL_PROXY=''
(Rekep) cindy@cindy-IdeaPad-Linux:~/Documents/R2S2R/Enhanced_ReKep4xarm_Tinker$ python main_rekep.py --use_cached_query
```

## Visualization assistance
ros2 run tf2_ros static_transform_publisher --x -0.09537 --y -0.4684 --z 0.397804 --qx -0.004694 --qy 0.267570 --qz 0.192732 --qw 0.944054 --frame-id link_base --child-frame-id camera_link

ros2 run tf2_ros static_transform_publisher --x 0.7157 --y 0.6381 --z 0.5705 --qx 0.256298 --qy 0.263371 --qz -0.646286 --qw 0.668776 --frame-id link_base --child-frame-id camera_link
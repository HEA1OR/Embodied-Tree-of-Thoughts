# 2. Point Tracker Part --READ ME FIRST PLZ
1. Install Tapnet following [tapnet](https://github.com/google-deepmind/tapnet)
2. Put the `point_track_ros.py` under [tapnet/tapnet](https://github.com/google-deepmind/tapnet/tree/main/tapnet)
3. Make sure the camera has been launched (just once is ok)
``````bash
roslaunch realsense2_camera rs_camera.launch
```
4. Open the project root folder of [tapnet](https://github.com/google-deepmind/tapnet)
```bash
export PYTHONPATH=`(cd ../ && pwd)`:`pwd`:$PYTHONPATH
python ./tapnet/point_track_ros.py
```
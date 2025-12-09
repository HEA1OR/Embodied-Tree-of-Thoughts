# 3. Anygrasp part--READ ME FIRST

1. Please install `anygrasp_sdk` following [anygrasp](https://github.com/graspnet/anygrasp_sdk)
2. Put the `detecter_ros.py` under [grasp_detection](https://github.com/graspnet/anygrasp_sdk/tree/main/grasp_detection)
3. Launch the camera 
```bash
roslaunch realsense2_camera rs_camera.launch
```
4. Open the project root folder of `anygrasp_sdk`
```bash
cd grasp_detection
python detecter_ros.py
```

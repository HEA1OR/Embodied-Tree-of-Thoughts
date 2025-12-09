# 1. Filter the target objects' masks
This code is adapted from [copa](https://github.com/HaoxuHuang/copa)
1. Create a new root folder, and put `grasp` & `som_gpt4v`in it.
2. Install environment following [copa](https://github.com/HaoxuHuang/copa) (Just the environment, not code!)
3. Change the task instruction in `SoM/grasp/grasp_cfg.yaml`
4. Make sure the camera has been launched (just once is ok)
``````bash
roslaunch realsense2_camera rs_camera.launch
```
4. Open in the root folder, run: 
```bash
python som_gpt4v/main_som.py grasp
```
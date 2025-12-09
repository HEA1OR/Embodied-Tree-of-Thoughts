# Enhanced Rekep for xArm6

## Introduction

这是一个基于[Rekep](https://github.com/huangwl18/ReKep) 与 [Koch_VLM_Benchmarks](https://github.com/Quest2GM/Koch_VLM_Benchmarks) 改进，增加了一些新的功能，并专门部署在[xArm6](https://www.ufactory.cc/xarm-collaborative-robot/)机械臂上的项目。

## TODO&CHANGED List

基于原仓库做出以下改变/补充，代码将陆续完善更新（预计6月前），请持续关注！：

- [x] 增加[SoM](https://github.com/microsoft/SoM) 预先筛选目标操作物体mask，以减少关键点数量

- [x] 增加 [Tapnet](https://github.com/google-deepmind/tapnet) 追踪关键点

- [x] 增加了抓取检测模型模块[Anygrasp](https://github.com/graspnet/anygrasp_sdk)

- [ ] 使用[GenPose++](https://github.com/Omni6DPose/GenPose2)增加关键矢量的追踪，参考了[Omnimanip](https://github.com/pmj110119/OmniManip) 论文，部分代码可以参考我的另一个仓库[GenPose2-SAM-2-real-time](https://github.com/youngfriday/GenPose2-SAM-2-real-time)

- [ ] 使用 [SAM2]()进行mask的追踪

## A Simple Video

- ***TASK Instruction:*** **pick up the red block and drop it into the box**

> 为了节省时间与运算空间：此视频中没有使用`path solver`，所以第二个转换过程略显不自然

https://github.com/user-attachments/assets/512991e5-2401-4ae1-913f-e09249200794https://github.com/user-attachments/assets/512991e5-2401-4ae1-913f-e09249200794

## To Prepare

- ***通讯器：*** 为了尽可能满足各个模块的环境，使用 [ROS](https://www.ros.org/) 进行通讯，尤其是向多个模块传输图像
- ***xArm6 控制器：*** 本项目使用 [xArm-Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK)
- ***RGB-D 相机：*** realsense d435i，使用 [realsense-ros](https://github.com/IntelRealSense/realsense-ros) 发布图像。同时注意提前将你的相机外参矩阵（SE(3)4x4）保存为`camera_extrinsic1.npy` 放到根目录中。
- ***系统与显卡：*** 目前在一台Ubuntu 20.04上运行，配备一台10GB的Nvidia 3080显卡，为了防止显存溢出，我选择将 [SoM] 部分提前分开运行。 
- ***环境：*** 各个子模块的环境，请严格依照原项目配置，最终同时分开运行
- ***CUDA：*** 12.1

## Reference

- [Rekep](https://github.com/huangwl18/ReKep) 
- [Koch_VLM_Benchmarks](https://github.com/Quest2GM/Koch_VLM_Benchmarks) 
- [Omnimanip](https://github.com/pmj110119/OmniManip) 
- [Copa](https://github.com/HaoxuHuang/copa)



 
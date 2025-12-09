
<h2 align="center"><a href="" >
Embodied Tree of Thoughts: Deliberate Manipulation Planning with Embodied World Model</a></h2>


<h5 align="center">

<a href="https://arxiv.org/abs/XXXX.XXXXX">
  <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg" alt="arXiv">
</a>

<a href="https://embodied-tree-of-thoughts.github.io/">
  <img src="https://img.shields.io/badge/Website-embodied--tree--of--thoughts.github.io-9C276A.svg" alt="Website">
</a>

</h5>


<div align="center"><video src="https://github.com/user-attachments/assets/9255280d-279f-4b62-b861-947b4ffd1927" width="800" autoplay loop muted></div>

## 🌟 Introduction

We propose **Embodied Tree of Thoughts (EToT)**, a novel Real2Sim2Real planning framework that utilizes a reconstructed simulation environment as a world model for tree-search-based task planning.

<div style="text-align: center;">
  <img src="static/etot.png" style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
</div>
<br>
This codebase is organized into four main components:

1. **Simulation Experiments**  
2. **VLM-based Task Planning**  
3. **Real-World Robot Experiments**  
4. **Scene Reconstruction**
---

## Installation

> **Notice:** This project is <span style="color: red; background-color: yellow;">Under Continuous Update</span>. 


### 1. Simulation Experiments

We provide **all simulation code and digital assets used in the paper.**  
For installation and usage instructions, please refer to  
👉 **[README_SIM.md](./simworld/README_SIM.md)**



### 2. VLM Task Planning

This repository includes the full implementation of **EToT** as well as all baseline task-planning methods used in the paper.  
For details on how to run and configure the planners, see  
👉 **[README_VLM.md](./vlm/README_VLM.md)**



### 3. Real-World Experiments

Real-robot execution requires adapting the code to your robot and camera hardware.  
We provide an example setup using **xArm6 + Azure Kinect**, which you can modify to match your own robot and sensor configuration.

Installation and execution instructions are available in  
👉 **[README_REAL.md](./realworld/README_REAL.md)**


### 4. Scene Reconstruction

The reconstruction pipeline builds on **[SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects)**, **[FoundationPose](https://github.com/NVlabs/FoundationPose)**, and **[DexSim2Real<sup>2</sup>](https://github.com/jiangtaoran/DexSim2Real2)**.  

Detailed instructions can be found in  
👉 **[README_REAL2SIM.md](./r2s/README_REAL2SIM.md)**

---
![Short-horizon tasks(Task1-4, Disturbance)](static/task1.png)
![Long-horizon tasks(Task5-7)](static/task2.png)

We have publicly released experimental videos for all tasks on [Google Drive](https://drive.google.com/file/d/1IWaRJY0s3qXccU_GSkYx_LwXhquxBIZ3/view?usp=sharing), including simulator views, camera footage, and third-person camera recordings.

> This repository is built on top of:
> - [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K)
> - [ReKep](https://github.com/huangwl18/ReKep)
> - [Enhanced_ReKep4xarm](https://github.com/youngfriday/Enhanced_ReKep4xarm/)
> - *and several other open-source repositories.*

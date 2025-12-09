
## Simulation Environment Setup

The experiment requires two separate environments:  
- **simworld** – for the simulation engine  
- **anygrasp** – for grasp-pose computation  

We recommend installing them in **isolated** conda environments.

--------------------------------------------------
### 1. Main Environment (simworld)

1. Create a conda env with **Python 3.10**, PyTorch and NumPy:

```bash
conda create -n simworld python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 "numpy<2" -c pytorch -c nvidia
conda activate simworld
```

2. Clone the Embodied-Tree-of-Thoughts repo:
```bash
git clone https://github.com/hea1or/Embodied-Tree-of-Thoughts.git
```

3. Enter the simworld folder and clone [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) (we need `v1.1.1` and will rename it to OmniGibson):
```bash
cd Embodied-Tree-of-Thoughts/simworld
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git OmniGibson
cd OmniGibson
git checkout v1.1.1
```
4. Install OmniGibson in source editable mode:
```bash
pip install -e .
```
5. Run the installation script to download Isaac Sim, the OmniGibson dataset and all default assets:

```shell
python -m omnigibson.install
```
#### Additional Asset Download (required for tasks)
We provide an extra asset pack, which includes object assets and robot models in the experiment. After unpacking you will get:
```
assets/
├── assets/
└── robot/
    ├── xarm6.py
    └── xarm6/
```
1. In the Embodied-Tree-of-Thoughts folder, download and unzip it  with:
```bash
gdown https://drive.google.com/uc?id=1_bczNtle39hcxT3IYA2L98-W2rURBz2K -O assets.zip && unzip assets.zip -d assets && rm assets.zip
```

2. Move the inner assets folder into simworld:
```
cp -r assets/assets simworld/
```
3. Register the xarm6 robot (simplified version of the [official guide](https://behavior.stanford.edu/tutorials/custom_robot_import.).)
If you followed our steps (OmniGibson installed from source), its asset root is
simworld/OmniGibson/omnigibson/data/assets/.
If you installed OmniGibson via pip, the path is inside your conda folder – locate it yourself.
```
# copy robot model
cp -r assets/robot/xarm6 simworld/OmniGibson/omnigibson/data/assets/models/

# copy robot config
cp assets/robot/xarm6.py simworld/OmniGibson/omnigibson/robots/
```
4. Finally, add the following line at the end of the `simworld/OmniGibson/omnigibson/robots/__init__.py` file:
```python
from omnigibson.robots.xarm6 import xarm6
```

>Due to inconsistencies in the directory structures of different programs and simulators, we use absolute paths in certain parts of the code to ensure stable execution. For user convenience, these paths can be updated by adjusting the configuration and running:
>```bash
>python simworld/xarm6_tasks/path_cover.py
>```
>This script replaces all hard-coded paths across the project. To ensure full coverage, we recommend additionally searching for the string:`/home/dell/workspace/xwj`and replace it with your own path.


After installation, you can run:

```bash
python simworld/OmniGibson/omnigibson/examples/robots/robot_control_example.py
```

and select **xarm6** to verify whether it can move properly.

A minimal demo is supplied; you can run it before installing anygrasp.
```bash
python simworld/env_build/task2_demo.py
```
### 2. Anygrasp Environment


To execute the full pipeline you need the anygrasp SDK. Please follow the official installation instructions at [anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk). We create a dedicated Conda environment named `anygrasp`, and the license and related files should be placed inside the corresponding `simworld/anygrasp` directory.

In our implementation, line 76 of `simworld/xarm6_tasks/action_utils.py` invokes AnyGrasp via `subprocess`. You will need to modify the path according to your local setup, or rewrite this invocation using ROS or another interface. (Our real-world experiment code uses ROS-based invocation, which you may refer to as an example.)

After installation, you can directly run `anygrasp.py` to verify that the program works correctly: 
```bash
conda activate anygrasp
python simworld/anygrasp/anygrasp.py
```
Additionally, you may uncomment the code at line 131 to enable point-cloud visualization.

## Simulation Tasks 
We provide 8 manipulation tasks using the xarm6 robot in `simworld/xarm6_tasks/`. You can run them with:
```bash
conda activate simworld
python simworld/xarm6_tasks/main.py
```
You can follow the terminal prompts to select a task and one of the available plans. Plans marked with (feasible) indicate valid solutions. An example interaction is shown below:
```bash
(simworld) dell@dell-Precision-7960-Tower:~/workspace/xwj/Embodied-Tree-of-Thoughts/simworld/xarm6_tasks$ python main.py 
Tasks:
  0  ->  disturbance   (Pick up a tennis ball.)
  1  ->  short_1       (Open the door of the microwave oven.)
  2  ->  short_2       (Reorient the pen and drop it into a holder.)
  3  ->  short_3       (Pick up the holder horizontally or vertically.)
  4  ->  short_4       (Close the drawer.)
  5  ->  long_1        (Reorient the pen and drop it into a holder.)
  6  ->  long_2        (Put the apple and the pen holder on the drawer, and the apple should be placed in the pen holder.)
  7  ->  long_3        (Put the apple and the tennis ball in either the drawer or the pen holder, together or separately.)
Please input task number (0/1/2/3/4/5/6/7): 1

The plan meanings of task short_1: 
  plan 0  ->  (feasible) tennis>>safe, open the door
  plan 1  ->  only open the door
Please choose plan number (0/1) for task short_1: 0

Launch: /home/dell/miniconda3/envs/simworld/bin/python /home/dell/workspace/xwj/Embodied-Tree-of-Thoughts/simworld/xarm6_tasks/short_1.py --plan 0
---
```
---
>If you encounter any issues during the process above, please open an issue or contact us at 2604317843xwj@gmail.com. For simulator-related problems, you may also check the issue tracker of [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) for potential solutions.

 **This part of the repository is adapted from [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) and [ReKep](https://github.com/huangwl18/ReKep).**
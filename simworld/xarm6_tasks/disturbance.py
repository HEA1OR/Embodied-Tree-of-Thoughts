#!/usr/bin/env python3
import argparse
import sys
import os
import time
sys.path.append(os.path.join(os.path.dirname(__file__), "../rekep"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch as th
from rekep.main import Main
from action_utils import move_delta, grasp

def grasp_tennis(main, obj):
    grasp(main, obj, offset=np.array([0, 0, 0.13]), optimaize=True)
    move_delta(main, np.array([0, 0, 0.1]), ori=np.array([1, 0, 0, 0]))

def run(plan: int):
    scene_file = './xarm6_envs/disturbance.json'
    main = Main(scene_file, visualize=False)
    tennis1 = main.env.og_env.scene.object_registry("name", "tennis1")
    tennis2 = main.env.og_env.scene.object_registry("name", "tennis2")
    holder_1 = main.env.og_env.scene.object_registry("name", "holder_1")
    pos, ori = tennis1.get_position_orientation()
    if plan in (0, 1):
        holder_1.set_position_orientation(position=[pos[0]+1.4, pos[1]+0.2, pos[2]], orientation=ori)
        time.sleep(1)
    if plan == 0:
        grasp_tennis(main, tennis1)
    elif plan == 1:
        grasp_tennis(main, tennis2)
    elif plan == 2:
        grasp_tennis(main, tennis1)
    elif plan == 3:
        grasp_tennis(main, tennis2)
    else:
        raise ValueError("plan must be 0,1,2,3")
    main.env.save_video()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=int, choices=[0, 1, 2, 3], required=True)
    args = parser.parse_args()
    try:
        run(args.plan)
    except Exception as e:
        print(e)
        sys.exit(1)
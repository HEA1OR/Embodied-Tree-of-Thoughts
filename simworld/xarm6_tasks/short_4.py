#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../rekep"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch as th
from rekep.main import Main
from action_utils import move_delta, move_to, grasp

def grasp_toy(main, toy):
    main._execute_release_action()
    grasp(main, toy, offset=np.array([-0.01, 0., 0.13]), optimaize=True)
    move_delta(main, np.array([0.0, 0.0, 0.10]), fast=False)
    move_delta(main, np.array([0.09, 0.11, 0.05]), fast=False)
    move_delta(main, np.array([0.0, 0.0, -0.10]), fast=False)
    main._execute_release_action()
    move_delta(main, np.array([0.0, 0.0, 0.10]), fast=False)
    main.env.robot._controllers['gripper_0']._open_qpos = th.full((6,), 0.8)
    main.env.close_gripper()

def close(main):
    home_pose = np.array([0.3327, -0.0893, 1.3360, -0.9992, -0.0401, 0.0029, 0.0001])
    move_to(main, home_pose[:3], ori=home_pose[3:], fast=True)
    move_delta(main, np.array([0.0, 0.0, -0.10]), fast=True)
    move_delta(main, np.array([0.15, 0.15, 0.0]), fast=True)
    move_delta(main, np.array([-0.1, -0.1, 0.0]), fast=True)

def run(plan: int):
    scene_file = './xarm6_envs/short_4.json'
    main = Main(scene_file, visualize=False)
    toy = main.env.og_env.scene.object_registry("name", "toy")
    if plan == 0:
        close(main)
    elif plan == 1:
        grasp_toy(main, toy)
        close(main)
    else:
        raise ValueError("plan must be 0 or 1")
    main.env.save_video()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    try:
        run(args.plan)
    except Exception as e:
        print(e)
        sys.exit(1)
#!/usr/bin/env python3
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../rekep"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch as th
from rekep.main import Main
from action_utils import move_to, move_delta

def grasp_holder(main, holder):
    main._execute_release_action()
    target = np.array(holder.get_position_orientation()[0]) + np.array([0, 0.05, 0.13])
    move_to(main, target, fast=True)
    main.env.robot._controllers['gripper_0']._closed_qpos = th.full((6,), 0.2)
    main.env.robot._controllers['gripper_0']._open_qpos = th.full((6,), 0.8)
    main._execute_grasp_action()
    move_delta(main, np.array([0, 0, 0.15]), fast=True)
    drawer_pos = np.array(main.env.og_env.scene.object_registry("name", "drawer").get_position_orientation()[0])
    move_to(main, drawer_pos + np.array([0, 0.1, 0.20]), fast=True)
    move_delta(main, np.array([0, 0, -0.06]), fast=True)
    main._execute_release_action()
    move_delta(main, np.array([0, 0, 0.12]), fast=True)

def grasp_apple(main, apple):
    main._execute_release_action()
    pos = np.array(apple.get_position_orientation()[0]) + np.array([0.0, 0.33, -0.05])
    move_to(main, pos, fast=True)
    main.env.robot._controllers['gripper_0']._closed_qpos = th.full((6,), 0.0)
    main.env.robot._controllers['gripper_0']._open_qpos = th.full((6,), 0.2)
    main._execute_grasp_action()
    move_delta(main, np.array([0, 0, 0.2]), fast=True)

def drop_in(main, target):
    target_pos = np.array(target.get_position_orientation()[0])
    move_to(main, target_pos + np.array([0, 0, 0.2]), fast=True)
    move_delta(main, np.array([0, 0, -0.1]), fast=True)
    main._execute_release_action()
    move_delta(main, np.array([-0.08, 0, 0.08]), fast=True)

def run(plan: int):
    scene_file = './xarm6_envs/long_2.json'
    main = Main(scene_file, visualize=False)
    holder_1 = main.env.og_env.scene.object_registry("name", "holder_1")
    apple = main.env.og_env.scene.object_registry("name", "apple")
    if plan == 0:
        grasp_holder(main, holder_1)
        grasp_apple(main, apple)
        drop_in(main, holder_1)
    elif plan == 1:
        grasp_apple(main, apple)
        drop_in(main, holder_1)
        grasp_holder(main, holder_1) # or grasp_apple(main, apple)
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
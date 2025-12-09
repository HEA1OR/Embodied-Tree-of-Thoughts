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
from scipy.spatial.transform import Rotation as R

def open_drawer(main):
    main._execute_release_action()
    move_to(main, np.array([0.4756, 0.1068, 1.3650]), fast=True)
    move_delta(main, np.array([0.0, 0.0, -0.10]), fast=True)
    move_delta(main, np.array([0.07, 0.11, 0.0]), fast=True)
    move_delta(main, np.array([-0.09, -0.04, 0.20]), fast=True)

def close_drawer(main):
    move_to(main, np.array([0.5456, 0.2568, 1.3650]), fast=True)
    move_delta(main, np.array([0.0, 0.0, -0.10]), fast=True)
    move_delta(main, np.array([-0.067, -0.15, 0.0]), fast=True)

def grasp_apple(main, apple):
    main._execute_release_action()
    tgt = np.array(apple.get_position_orientation()[0]) + np.array([0.0, 0.33, -0.05])
    move_to(main, tgt, fast=True)
    main.env.robot._controllers['gripper_0']._closed_qpos = th.full((6,), 0.0)
    main.env.robot._controllers['gripper_0']._open_qpos = th.full((6,), 0.2)
    main._execute_grasp_action()
    move_delta(main, np.array([0.0, 0.0, 0.2]), fast=True)

def grasp_tennis1(main, tennis):
    grasp(main, tennis, offset=np.array([0.01, 0.01, 0.08]), optimaize=True, fast=False)
    move_delta(main, np.array([0.05, 0.05, 0.32]), fast=False)
    move_delta(main, np.array([0.0, 0.0, 0.1]), fast=False)

def grasp_tennis2(main, tennis):
    grasp(main, tennis, offset=np.array([-0.02, 0.0, 0.13]), optimaize=True, fast=True)
    move_delta(main, np.array([0.0, 0.0, 0.15]), fast=True)

def drop_in(main, target):
    pos = np.array(target.get_position_orientation()[0])
    move_to(main, pos + np.array([0.0, 0.0, 0.2]), fast=True)
    move_delta(main, np.array([0.0, 0.0, -0.07]), fast=True)
    main._execute_release_action()
    move_delta(main, np.array([-0.08, 0.0, 0.08]), fast=True)

def place_on(main, target):
    pos = np.array(target.get_position_orientation()[0])
    move_to(main, pos + np.array([-0.06, -0.1, 0.20]), fast=True)
    move_delta(main, np.array([0.0, 0.0, -0.10]), fast=True)
    main._execute_release_action()
    move_delta(main, np.array([0.0, 0.0, 0.12]), fast=True)

def drop_in_drawer(main):
    move_to(main, np.array([0.5, 0.125, 1.4376]), fast=True)
    move_delta(main, np.array([0.0, 0.0, -0.10]), fast=True)
    main._execute_release_action()
    move_delta(main, np.array([-0.04, 0.0, 0.1]), fast=True)

def run(plan: int):
    scene_file = './xarm6_envs/long_3.json'
    main = Main(scene_file, visualize=False)
    holder_1 = main.env.og_env.scene.object_registry("name", "holder_1")
    apple = main.env.og_env.scene.object_registry("name", "apple")
    drawer = main.env.og_env.scene.object_registry("name", "drawer")
    tennis = main.env.og_env.scene.object_registry("name", "tennis1")
    if plan == 0:
        grasp_apple(main, apple)
        drop_in(main, holder_1)
        grasp_tennis1(main, tennis)
        place_on(main, drawer)
        open_drawer(main)
        grasp_tennis2(main, tennis)
        drop_in_drawer(main)
        close_drawer(main)
    elif plan == 1:
        grasp_apple(main, apple)
        drop_in(main, holder_1)
        grasp_tennis1(main, tennis)
        drop_in(main, holder_1)
    elif plan == 2:
        grasp_apple(main, apple)
        drop_in(main, holder_1)
        open_drawer(main)
        grasp_tennis1(main, tennis)
    elif plan == 3:
        open_drawer(main)
        grasp_apple(main, apple)
        drop_in_drawer(main)
        close_drawer(main)
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
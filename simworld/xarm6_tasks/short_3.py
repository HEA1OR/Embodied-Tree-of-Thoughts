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


def sg1(main, obj):
    return np.concatenate([
        np.array(obj.get_position_orientation()[0]) + np.array([0.0, -0.05, 0.13]),
        main.env.get_ee_pose()[3:]
    ])

def sg2(main, obj):
    return np.concatenate([
        np.array(obj.get_position_orientation()[0]) + np.array([0.0, -0.13, 0.035]),
        np.array([-0.5, -0.5, -0.5, 0.5])
    ])

def grasp_vertical(main, obj, q_close=0.2, q_open=0.8):
    main._execute_release_action()
    move_to(main, sg1(main, obj)[:3], ori=sg1(main, obj)[3:], fast=True)
    main.env.robot._controllers['gripper_0']._closed_qpos = th.full((6,), q_close)
    main.env.robot._controllers['gripper_0']._open_qpos   = th.full((6,), q_open)
    main._execute_grasp_action()
    move_delta(main, np.array([0.0, 0.0, 0.1]), fast=True)

def grasp_horizontal(main, obj, q_close=0.0, q_open=0.3):
    main._execute_release_action()
    move_to(main, sg1(main, obj)[:3], ori=sg1(main, obj)[3:], fast=True)
    main.env.robot._controllers['gripper_0']._closed_qpos = th.full((6,), q_close)
    main.env.robot._controllers['gripper_0']._open_qpos   = th.full((6,), q_open)
    main._execute_grasp_action()
    move_delta(main, np.array([0.0, 0.0, 0.1]), fast=True)

def run(plan: int):
    scene_file = './xarm6_envs/short_3.json'
    main = Main(scene_file, visualize=False,
                config_path="./rekep/configs/config_xarm6.yaml")
    holder = main.env.og_env.scene.object_registry("name", "holder_1")

    if plan == 0:
        grasp_vertical(main, holder)
    elif plan == 1:
        grasp_horizontal(main, holder)
    else:
        raise ValueError("plan must be 0 or 1")

    main.env.save_video()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=int, choices=[0, 1], required=True,
                        help="0: vertical grasp;  1: horizontal grasp")
    args = parser.parse_args()
    try:
        run(args.plan)
    except Exception as e:
        print(e)
        sys.exit(1)
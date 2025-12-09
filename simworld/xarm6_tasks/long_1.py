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

def grasp_pen(main, obj):
    main._execute_release_action()
    grasp(main, obj, offset=np.array([-0.02, 0, 0.1]), optimaize=True, fast=True)
    move_delta(main, np.array([0, 0, 0.1]), fast=True)

def grasp_apple(main, holder):
    main._execute_release_action()
    target = holder.get_position_orientation()[0] + np.array([0.0, 0.0, 0.203])
    move_to(main, np.array(target), fast=True)
    main.env.robot._controllers['gripper_0']._closed_qpos = th.full((6,), 0.0)
    main.env.robot._controllers['gripper_0']._open_qpos = th.full((6,), 0.2)
    main._execute_grasp_action()
    move_delta(main, np.array([0.0, 0.0, 0.1]), fast=True)
    move_delta(main, np.array([-0.15, 0.0, 0.0]), fast=True)
    move_delta(main, np.array([0.0, 0.0, -0.15]), fast=True)
    main._execute_release_action()
    move_delta(main, np.array([0.0, 0.0, 0.2]), fast=True)
    move_delta(main, np.array([0.0, -0.2, 0.0]), fast=True)

def drop_in(main, target):
    target_pos = target.get_position_orientation()[0]
    move_to(main, np.array(target_pos) + np.array([0, 0, 0.25]), fast=True)
    grasp_ori = main.env.get_ee_pose()[3:]
    mat = R.from_quat(grasp_ori).as_matrix()
    Rz = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    R180 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    new_quat = R.from_matrix(mat @ Rz @ R180).as_quat()
    move_to(main, main.env.get_ee_pose()[:3], ori=new_quat, fast=True)
    move_delta(main, np.array([0, 0, -0.1]), fast=True)
    main._execute_release_action()
    move_delta(main, np.array([-0.1, 0, 0.1]), fast=True)

def drop_in2(main, target):
    target_pos = target.get_position_orientation()[0]
    move_to(main, np.array(target_pos) + np.array([0, 0, 0.25]), fast=True)
    grasp_ori = main.env.get_ee_pose()[3:]
    mat = R.from_quat(grasp_ori).as_matrix()
    Rz = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    R180 = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    new_quat = R.from_matrix(mat @ Rz @ R180).as_quat()
    move_to(main, main.env.get_ee_pose()[:3], ori=new_quat, fast=True)
    main._execute_release_action()
    move_delta(main, np.array([-0.1, 0, 0.1]), fast=True)

def run(plan: int):
    scene_file = './xarm6_envs/long_1.json'
    main = Main(scene_file, visualize=False)
    pen = main.env.og_env.scene.object_registry("name", "pen")
    holder_1 = main.env.og_env.scene.object_registry("name", "holder_1")
    holder_2 = main.env.og_env.scene.object_registry("name", "holder_2")
    apple = main.env.og_env.scene.object_registry("name", "apple")
    if plan == 0:
        grasp_pen(main, pen)
        drop_in(main, holder_1)
    elif plan == 1:
        grasp_pen(main, pen)
        drop_in2(main, holder_2)
    elif plan == 2:
        grasp_apple(main, holder_2)
        grasp_pen(main, pen)
        drop_in(main, holder_2)
    else:
        raise ValueError("plan must be 0, 1 or 2")
    main.env.save_video()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=int, choices=[0, 1, 2], required=True)
    args = parser.parse_args()
    try:
        run(args.plan)
    except Exception as e:
        print(e)
        sys.exit(1)
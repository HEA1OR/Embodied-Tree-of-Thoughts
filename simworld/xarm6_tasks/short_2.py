#!/usr/bin/env python3

import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../rekep"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rekep.main import Main
import numpy as np
import torch as th
from scipy.spatial.transform import Rotation as R
from action_utils import grasp, move_delta, move_to


def grasp_pen(main, obj):
    grasp(main, obj, offset=np.array([-0.01, 0, 0.13]), optimaize=True)
    move_delta(main, pose_delta=np.array([0, 0, 0.1]), ori=np.array([1, 0, 0, 0]))


def drop_in(main, target):
    target_position = target.get_position_orientation()[0]
    cur_pose = main.env.get_ee_pose()
    next_subgoal = cur_pose.copy()
    next_subgoal[:3] = target_position + np.array([0, 0, 0.25])
    grasp_ori = next_subgoal[3:]
    grasp_mat = R.from_quat(grasp_ori).as_matrix()

    Rz = np.array([[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]])
    rotated_mat = grasp_mat @ Rz
    R_z_180 = np.array([[-1, 0, 0],
                        [0, -1, 0],
                        [0, 0, 1]])
    rotated_mat = rotated_mat @ R_z_180
    next_subgoal[3:] = R.from_matrix(rotated_mat).as_quat()

    move_to(main, next_subgoal[:3], next_subgoal[3:], fast=True)
    move_delta(main, np.array([0, 0, -0.1]))
    main._execute_release_action()
    move_delta(main, np.array([-0.1, 0, 0.1]))


def run(plan: int):
    scene_file = './xarm6_envs/short_2.json'
    main = Main(scene_file, visualize=False)
    pen = main.env.og_env.scene.object_registry("name", "pen")
    holder_1 = main.env.og_env.scene.object_registry("name", "holder_1")
    holder_2 = main.env.og_env.scene.object_registry("name", "holder_2")

    if plan == 0:          # pen → holder_1
        grasp_pen(main, pen)
        drop_in(main, holder_1)
    elif plan == 1:        # pen → holder_2
        grasp_pen(main, pen)
        drop_in(main, holder_2)
    else:
        raise ValueError("plan must be 0 or 1")

    main.env.save_video()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=int, choices=[0, 1], required=True,
                        help="0: pen→holder_1;  1: pen→holder_2")
    args = parser.parse_args()
    try:
        run(args.plan)
    except Exception as e:
        print(e)
        sys.exit(1)
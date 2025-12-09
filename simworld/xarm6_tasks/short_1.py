import sys
sys.path.append("./rekep")
sys.path.append("../")
from rekep.main import Main
import numpy as np
import torch as th
from action_utils import get_subgoal, grasp, move_delta, move_to


def place_safe(main):
    move_delta(main, np.array([0.05, -0.23, 0.05]), ori=np.array([1, 0, 0, 0]))
    main._execute_release_action()
    move_delta(main, np.array([-0.05, 0, 0.1]), ori=np.array([1, 0, 0, 0]))
    main.env.robot._controllers['gripper_0']._open_qpos =  th.full((6,), 0.8)
    main._execute_grasp_action()


def open(main):
    next_subgoal_init = np.concatenate([[0.4588, 0.3084, 1.4023], [-0.9886, -0.0048, -0.1506, -0.0010]])
    next_subgoal = next_subgoal_init.copy() 
    next_subgoal[:3] += np.array([-0.15, 0, 0.2])
    move_to(main, next_subgoal[:3], next_subgoal[3:], fast=True)
    move_delta(main, np.array([0.15, 0, -0.2]), fast=True)
    move_delta(main, np.array([-0.3, -0.3, 0.0]), fast=True)
    

if __name__ == '__main__':

    scene_file = './xarm6_envs/short_1.json'
    try:
        main = Main(scene_file, visualize=False)
        tennis1 = main.env.og_env.scene.object_registry("name", "tennis1")
        plan = 0  # 0:pickplace and open; 1:only open
        if plan == 0:
            grasp(main, tennis1, offset=np.array([-0.01, 0, 0.13]), optimaize = True)
            place_safe(main)
            open(main)
        elif plan == 1:
            open(main)
        main.env.save_video()
    except Exception as e:
        print(e)


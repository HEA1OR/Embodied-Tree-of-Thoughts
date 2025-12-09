
from scipy.spatial.transform import Rotation as R
import numpy as np
from scipy.optimize import minimize
import imageio
import transform_utils as T
import subprocess
import torch as th

def rotate_around_y(grasp_ori, theta):
    grasp_mat = R.from_quat(grasp_ori).as_matrix()
    Ry = np.array([
        [np.cos(theta[0]), 0, np.sin(theta[0])],
        [0, 1, 0],
        [-np.sin(theta[0]), 0, np.cos(theta[0])]
    ])
    rotated_mat = grasp_mat @ Ry
    rotated_ori = R.from_matrix(rotated_mat).as_quat()
    return rotated_ori

def rotate_around_z(grasp_ori, theta):
    grasp_mat = R.from_quat(grasp_ori).as_matrix()
    Rz = np.array([
        [1, 0, 0],
        [0, np.cos(theta[0]), -np.sin(theta[0])],
        [0, np.sin(theta[0]), np.cos(theta[0])]
    ])
    rotated_mat = grasp_mat @ Rz
    rotated_ori = R.from_matrix(rotated_mat).as_quat()
    return rotated_ori

def objective_function(theta, grasp_ori, target_x):
    rotated_ori = rotate_around_y(grasp_ori, theta)
    rotated_x = R.from_quat(rotated_ori).apply([0, 0, 1])
    angle = np.arccos(np.clip(np.dot(rotated_x, target_x), -1.0, 1.0))
    return angle

def objective_function_z(theta, grasp_ori, target_z):
    rotated_ori = rotate_around_z(grasp_ori, theta)
    rotated_z = R.from_quat(rotated_ori).apply([0, 0, 1])
    angle = np.arccos(np.clip(np.dot(rotated_z, target_z), -1.0, 1.0))
    return angle

def optimize_rotation(grasp_ori, target_x):
    initial_theta = 0.0
    result = minimize(objective_function, initial_theta, args=(grasp_ori, target_x), bounds=[(-np.pi, np.pi)])
    optimal_theta = result.x
    optimal_ori = rotate_around_y(grasp_ori, optimal_theta)
    return optimal_ori

def optimize_rotation_z(grasp_ori, target_z):
    initial_theta = 0.1
    result = minimize(objective_function_z, initial_theta, args=(grasp_ori, target_z), bounds=[(-np.pi, np.pi)])
    optimal_theta = result.x
    optimal_ori = rotate_around_z(grasp_ori, optimal_theta)
    return optimal_ori


def get_subgoal(main, object, offset, optimaize=False):
    extrinsics = main.env.cams[0].extrinsics
    np.save('../anygrasp/extrinsics.npy', extrinsics)
    intrinsics = main.env.cams[0].intrinsics
    np.save('../anygrasp/intrinsics.npy', intrinsics)
    cam_obs = main.env.get_cam_obs()
    rgb = cam_obs[0]['rgb']
    depth = cam_obs[0]['depth']
    depth = depth * 1000
    depth = depth.numpy().astype(np.uint16)
    rgb_save_path = "../anygrasp/rgb.png"
    depth_save_path = "../anygrasp/depth.png"
    imageio.imwrite(rgb_save_path, rgb)
    imageio.imwrite(depth_save_path, depth)
    xyz = object.get_position_orientation()[0] + offset
    np.save('../anygrasp/target_point.npy', xyz)
    result = subprocess.run(
        ["conda", "run", "-n", "anygrasp", "python", "/home/dell/workspace/xwj/simworld/anygrasp/anygrasp.py"],  # You can also use ROS to build anygrasp subscriber
        capture_output=True,
        text=True)
    target_pose = np.load('../anygrasp/target_pose.npy')
    grasp_position = target_pose[:3]
    grasp_orientation = target_pose[3:] 
    R_inv = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    from scipy.spatial.transform import Rotation as R
    q0_mat = R.from_quat(grasp_orientation).as_matrix()
    q02 = q0_mat @ R_inv
    R_z_180 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    grasp_ori = R.from_matrix(q02).as_quat()
    if optimaize:
        target_x = np.array([0, 0, -1])
        target_z = np.array([0, 0, -1])
        optimal_ori_y = optimize_rotation(grasp_ori, target_x)
        optimal_ori_z = optimize_rotation_z(optimal_ori_y, target_z)
    else:
        optimal_ori_z = grasp_ori
    optimal_ori_z = R.from_quat(optimal_ori_z).as_matrix()
    q03 = optimal_ori_z @ R_z_180
    ee_pose = main.env.get_ee_pose()
    ee_mat = R.from_quat(ee_pose[3:]).as_matrix()
    if np.linalg.norm(optimal_ori_z - ee_mat) < np.linalg.norm(q03 - ee_mat):
        grasp_ori = R.from_matrix(optimal_ori_z).as_quat()
    else:
        grasp_ori = R.from_matrix(q03).as_quat()

    next_subgoal = np.concatenate([grasp_position, grasp_ori])
    grasp_offset = np.array([0, 0, 0.01]) + np.array([0, 0, -0.08])
    subgoal_pose_homo = T.convert_pose_quat2mat(next_subgoal)
    next_subgoal[:3] += subgoal_pose_homo[:3, :3] @ grasp_offset
    return next_subgoal

def move_delta(main, pose_delta, ori = None, fast = False):
    cur_pose = main.env.get_ee_pose()
    next_subgoal = cur_pose.copy()
    next_subgoal[:3] += pose_delta
    if ori is not None:
        next_subgoal[3:] = ori
    if fast:
        next_path = main._get_next_path_fast(next_subgoal, from_scratch=True)
    else:
        next_path = main._get_next_path(next_subgoal, from_scratch=True)
    main.action_queue = next_path.tolist()
    count = 0
    while len(main.action_queue) > 0 and count < main.config['action_steps_per_iter']:
        next_action = main.action_queue.pop(0)
        precise = len(main.action_queue) == 0
        main.env.execute_action(next_action, precise=precise)
        count += 1

def move_to(main, pose, ori = None, fast = False):
    move_delta(main, pose - main.env.get_ee_pose()[:3], ori = ori, fast = fast)

def grasp(main, object, offset, optimaize = False, fast = False, angle = None, angle2 = None):
    main._execute_release_action()
    next_subgoal = get_subgoal(main, object, offset=offset, optimaize=optimaize)
    if fast:
        next_path = main._get_next_path_fast(next_subgoal, from_scratch=True)
    else:
        next_path = main._get_next_path(next_subgoal, from_scratch=True)
    main.action_queue = next_path.tolist()
    count = 0
    while len(main.action_queue) > 0 and count < main.config['action_steps_per_iter']:
        next_action = main.action_queue.pop(0)
        precise = len(main.action_queue) == 0
        main.env.execute_action(next_action, precise=precise)
        count += 1
    if len(main.action_queue) == 0:
        if angle is None:
            width = np.load('../anygrasp/width.npy')
            angle = 0.85 - np.arcsin((width*3.4))
            if angle - 0.5 > 0.0:
                angle2 = angle - 0.5
            else:
                angle2 = 0.0
        main.env.robot._controllers['gripper_0']._closed_qpos =  th.full((6,), angle2)
        main.env.robot._controllers['gripper_0']._open_qpos =  th.full((6,), angle)
        main._execute_grasp_action()
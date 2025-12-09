import torch
import numpy as np
import json
import os
import argparse
from environment import ReKepOGEnv
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
import time
import subprocess
import imageio
from omnigibson.robots.fetch import Fetch
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

class Main:
    def __init__(self, scene_file, visualize=False, config_path="./rekep/configs/config_xarm6.yaml"):
        global_config = get_config(config_path)
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize

        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        self.env = ReKepOGEnv(global_config['env'], scene_file, verbose=False)

        ik_solver = IKSolver(
            robot_description_path=self.env.robot.robot_arm_descriptor_yamls[self.env.robot.default_arm],
            robot_urdf_path=self.env.robot.urdf_path,
            eef_name=self.env.robot.eef_link_names[self.env.robot.default_arm],
            reset_joint_pos=self.env.reset_joint_pos,
            world2robot_homo=self.env.world2robot_homo,
        )

        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos)
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.env.reset_joint_pos)
        self.visualizer = Visualizer(global_config['visualizer'], self.env)


    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        #self.env.reset()
        cam_obs = self.env.get_cam_obs()
        print("get_cam_obs")
        rgb = cam_obs[self.config['vlm_camera']]['rgb']
        points = cam_obs[self.config['vlm_camera']]['points']
        mask = cam_obs[self.config['vlm_camera']]['seg']
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        if rekep_program_dir is None:
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask)
            print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
            if self.visualize:
                self.visualizer.show_img(projected_img)
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # ====================================
        # = execute
        # ====================================
        self._execute(rekep_program_dir, disturbance_seq)

        action = np.zeros(7)  
        action[6] = -0.01
        # xunhuan
        for i in range(100):
            _ = self.env._step(action=action)


    def _update_disturbance_seq(self, stage, disturbance_seq):
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def _execute(self, rekep_program_dir, disturbance_seq=None):
        # load metadata
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stages'] + 1)}
        print("program info load")
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        print("register_keypoints")
        # load constraints
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
        print("constraint_fns")
        # bookkeeping of which keypoints can be moved in the optimization
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1)
        while True:
            scene_keypoints = self.env.get_keypoint_positions()
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)  # first keypoint is always the ee
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        if violation > self.config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:   
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._update_stage(new_stage)
            else:
                # apply disturbance
                self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = get optimized plan
                # ====================================
                if self.last_sim_step_counter == self.env.step_counter:
                    print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                count = 0
                while len(self.action_queue) > 0 and count < self.config['action_steps_per_iter']:
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise)
                    count += 1
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # if completed, save video and return
                    if self.stage == self.program_info['num_stages']: 
                        self.env.sleep(2.0)
                        save_path_rgb, save_path_depth = self.env.save_video()
                        print(f"{bcolors.OKGREEN}Video saved to {save_path_rgb}\n\n{bcolors.ENDC}")
                        print(f"{bcolors.OKGREEN}Video saved to {save_path_depth}\n\n{bcolors.ENDC}")
                        return
                    save_path_rgb, save_path_depth = self.env.save_video()
                    print(f"{bcolors.OKGREEN}Video saved to {save_path_rgb}\n\n{bcolors.ENDC}")
                    print(f"{bcolors.OKGREEN}Video saved to {save_path_depth}\n\n{bcolors.ENDC}")
                    # progress to next stage
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        '''subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)'''
        
        extrinsics = self.env.cams[0].extrinsics
        print('extrinsics:', extrinsics)
        np.save('extrinsics.npy', extrinsics)
        intrinsics = self.env.cams[0].intrinsics
        print('intrinsics:', intrinsics)
        np.save('intrinsics.npy', intrinsics)
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[0]['rgb']
        depth = cam_obs[0]['depth']
        depth = depth * 1000
        depth = depth.numpy().astype(np.uint16)
        rgb_save_path = "rgb.png"
        depth_save_path = "depth.png"
        imageio.imwrite(rgb_save_path, rgb)
        imageio.imwrite(depth_save_path, depth)
        print("rgb.shape", rgb.shape)
        print("depth.shape", depth.shape)
        if self.is_grasp_stage:    
            print("self.keypoints", self.program_info['grasp_keypoints'][self.stage - 1])
            xyz = self.keypoints[self.program_info['grasp_keypoints'][self.stage - 1]+1]  
            # 保存为npy文件
            np.save('target_point.npy', xyz)
            print("target_point saved")
            result = subprocess.run(
                ["conda", "run", "-n", "anygrasp", "python", "anygrasp.py"],
                capture_output=True,
                text=True)
            # 等anygrasp.py执行完成
           
            #读取target_pose.npy, 其中前三位是位置，后四位是四元数

            target_pose = np.load('target_pose.npy')
            grasp_position = target_pose[:3] # 假设位置在列表的前3个元素
            grasp_orientation = target_pose[3:] # 假设四元数在列表的后4个元素
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
            q03 = q02 @ R_z_180
            ee_pose = self.env.get_ee_pose()
            ee_mat = R.from_quat(ee_pose[3:]).as_matrix()
            # 选择q02和q03中与ee_mat最接近的一个
            if np.linalg.norm(q02 - ee_mat) < np.linalg.norm(q03 - ee_mat):
                grasp_ori = R.from_matrix(q02).as_quat()
            else:
                grasp_ori = R.from_matrix(q03).as_quat()
            
            #next_subgoal = np.concatenate([grasp_position, grasp_ori])
            next_subgoal = np.concatenate([grasp_position, np.array([1, 0, 0, 0])])
            grasp_offset = np.array([0, 0, -0.03])
            subgoal_pose_homo = T.convert_pose_quat2mat(next_subgoal)
            next_subgoal[:3] += subgoal_pose_homo[:3, :3] @ grasp_offset

            self.visualizer.visualize_subgoal(next_subgoal)

        else:
            print("getting next subgoal...")
            subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
            path_constraints = self.constraint_fns[self.stage]['path']
            next_subgoal, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                                self.keypoints,
                                                                self.keypoint_movable_mask,
                                                                subgoal_constraints,
                                                                path_constraints,
                                                                self.sdf_voxels,
                                                                self.collision_points,
                                                                self.is_grasp_stage,
                                                                self.curr_joint_pos,
                                                                from_scratch=from_scratch)
            subgoal_pose_homo = T.convert_pose_quat2mat(next_subgoal)
            print("subgoal_pose_______",next_subgoal)
            # if grasp stage, back up a bit to leave room for grasping
            debug_dict['stage'] = self.stage
            print_opt_debug_dict(debug_dict)
            if self.visualize:
                self.visualizer.visualize_subgoal(next_subgoal)
            print("Next subgoal1:", next_subgoal)

        return next_subgoal


    



    def _get_next_path(self, next_subgoal, from_scratch):
        self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
        self.collision_points = self.env.get_collision_points()
        self.curr_ee_pose = self.env.get_ee_pose()
        self.curr_joint_pos = self.env.get_arm_joint_postions()
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        print("path:", path)
        next_path = self._process_path(path)
        #if self.visualize:
        #self.visualizer.visualize_path(next_path)

        '''num_points = 10
        next_path = np.zeros((num_points, 8))
        goal_lin = np.linspace(self.env.get_ee_pose(), next_subgoal, num=num_points)
        next_path[:, :7] = goal_lin'''
        next_path[:, 7] = self.env.get_gripper_null_action()
        return next_path
    
    def _get_next_path_fast(self, next_subgoal, from_scratch):
        num_points = 3
        next_path = np.zeros((num_points, 8))
        goal_lin = np.linspace(self.env.get_ee_pose(), next_subgoal, num=num_points)
        next_path[:, :7] = goal_lin
        next_path[:, 7] = self.env.get_gripper_null_action()
        return next_path
    

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        print("num_steps:", num_steps)
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        print("dense_path:", dense_path)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([0, 0, 0.12])
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)
        
    
    def _execute_release_action(self):
        self.env.open_gripper()


    


import time
import numpy as np
import os
import rclpy
from std_msgs.msg import Int32MultiArray
from scipy.spatial.transform import Rotation as R
# from endeffector import EndEffector
from rekep.utils import (
    bcolors,
    get_clock_time,
    angle_between_rotmat,
    angle_between_quats,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
)
import re
def point_tracking_callback(msg):
    # 解析消息中的数据
    global tracking_points,point_idx,POINTS_CALLBACK
    tracking_points = []
    point_idx = []
    data = msg.data  # 获取传递的数据
    # 数据以每3个元素为一组（i, x, y），所以需要步长为3来提取
    for i in range(0, len(data), 3):
        point_index = data[i]  # 索引
        x = data[i + 1]        # x坐标
        y = data[i + 2]        # y坐标
        tracking_points.append((point_index, x, y))
        point_idx.append(point_index)
        # rospy.loginfo(f"Received point {point_index}: ({x}, {y})")
        POINTS_CALLBACK=True

    

class ReKepEnv:
    def __init__(self, config, robot, camera, endeffector, node=None, verbose=False):
        self.video_cache = []
        self.config = config
        self.verbose = verbose
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
        self.endeffector = endeffector
        self.robot = robot
        self.camera = camera
        self.node = node  # ROS2 node for subscriptions
        
        global tracking_points, point_idx, POINTS_CALLBACK
        POINTS_CALLBACK = False
        tracking_points = []
        point_idx = []
        
        gripper_pos = self.robot.get_gripper_position()
        if gripper_pos[1] > 700:
            self.gripper_state = int(0)
        else:
            self.gripper_state = int(1)

    # ======================================
    # = exposed functions
    # ======================================

    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        """Get signed distance field"""
        print("Getting SDF voxels (mock data)")
        # Return mock SDF grid
        sdf_voxels = np.zeros((10, 10, 10))
        return sdf_voxels


    def _get_obj_idx_dict(self, task_dir):
        
        # save prompt
        with open(os.path.join(task_dir, 'output_raw.txt'), 'r') as f:
            self.prompt = f.read()

        # Extract keypoint-object associations from comments
        matches = re.findall(r'(\w+)\s*\(keypoint\s*(\d+)\)', self.prompt)
        # matches = re.findall(r'((?:small|large) (?:red|blue|black) cube)\s*\(keypoint\s*(\d+)\)', self.prompt)
        # matches = re.findall(r'(white king|black king|box)\s*\(keypoint\s*(\d+)\)', self.prompt)

        # Generate dictionary dynamically
        keypoint_objects = {int(kp): obj.replace(" ", "_") for obj, kp in matches}

        return keypoint_objects

    def register_keypoints(self, keypoints, camera, rekep_dir):
        """
        Args:
            keypoints (np.ndarray): keypoints in the world frame of shape (N, 3)
        Returns:
            None
        Given a set of keypoints in the world frame, this function registers them so that their newest positions can be accessed later.
        i.e. Associate keypoints with their respective objects. We technically only need to pay attention to objects that will move in the future.
        """
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)

        self.keypoints = keypoints
        self._keypoint_registry = dict()

        idx_obj_map = self._get_obj_idx_dict(rekep_dir)
        print("idx_obj_map", idx_obj_map)

        # Set keypoint indices using index map from gpt-4o
        for i, k in enumerate(self.keypoints):
            obj = "none"
            if i in idx_obj_map.keys():
                obj = idx_obj_map[i]

            img_coord = camera.world_to_pixel_coordinates(k)
            # img_coord = self.endeffector.convert_world_to_point(camera, k)
            print(f"Keypoint {i}: Object: {obj}, World Coord: {k}, Image Coord: {img_coord}")
            self._keypoint_registry[i] = {"object": obj, 
                                          "keypoint": k,
                                          "img_coord": img_coord,
                                          "on_grasp_coord": None,
                                          "is_grasped": False}

    def get_keypoint_positions(self):
        """
        Args:
            None
        Returns:
            np.ndarray: keypoints in the world frame of shape (N, 3)
        Given the registered keypoints, this function returns their current positions in the world frame.
        Keypoints are updated by taking the keypoints and updating them if the gripper is holding them or not.
        """
        global tracking_points, point_idx,POINTS_CALLBACK
        keypoint_positions = []
        # 订阅ros节点消息
        # 订阅/当前追踪点的位置消息
        if not POINTS_CALLBACK and self.node is not None:
            self.tracking_sub = self.node.create_subscription(
                Int32MultiArray, '/current_tracking_points', 
                point_tracking_callback, 10)
            # 等待回调函数填充数据
            while not tracking_points or not point_idx:
                if self.node:
                    self.node.get_logger().info("Waiting for tracking points to be updated...")
                rclpy.spin_once(self.node, timeout_sec=3.0)

        
        print("tracking_points", tracking_points)
        print("point_idx", point_idx)
        for idx, obj in self._keypoint_registry.items():
        
            # if obj["is_grasped"]:
                
                # # Determine difference between last ee_pose and grip_pose
                # init_ee_point = obj["on_grasp_coord"]
                # _, ee_point = self.endeffector.return_estimated_ee(self.camera, self.get_ee_pos())
                # diff_point = [ee_point[0] - init_ee_point[0], ee_point[1] - init_ee_point[1]]
                
                # # Update curr_pose of object
                # obj_point = obj["img_coord"]
                # curr_point = [obj_point[0] + diff_point[0], obj_point[1] + diff_point[1]]
                # obj["img_coord"] = list(curr_point)
                # obj["on_grasp_coord"] = list(ee_point)
                # _, ee_point = self.endeffector.return_estimated_ee(self.camera, self.get_ee_pos())
                # obj["img_coord"] = list(ee_point)
                
            # Convert updated coord to camera
            # obj["keypoint"] = self.endeffector.find_closest_point_to_world(obj["img_coord"])
            

            # NOTE: 只更新接收到的点，无关点填0
            if idx not in point_idx:
                obj['img_coord']= [0, 0]
                obj['keypoint']= self.keypoints[idx]  
                keypoint_positions.append(obj["keypoint"])
                self.node.get_logger().info(f"Keypoint {idx} not in tracked points, using previous keypoint: {obj['keypoint']}")
                continue
            # debug
            print("idx", idx)
            # 找到 tracking_points 中对应 idx 的元素
            matching_point = next((point for point in tracking_points if point[0] == idx), None)
            if matching_point is None:
                print(f"Warning: No matching point found for idx {idx}")
                continue

            print("tracking_points[idx][1:3]", matching_point[1:3])
            obj['img_coord'] = matching_point[1:3]
            # visualize point on camera, display it directly
            # import cv2
            # rgb_image = self.camera.capture_image("rgb")
            # cv2.circle(rgb_image, (obj['img_coord'][0], obj['img_coord'][1]), 5, (0, 255, 0), -1)
            # cv2.imshow("Keypoint Tracking", rgb_image)
            # cv2.waitKey(1)

            print("converting to world coordinates...")
            self.node.get_logger().info(f"Converting to world coordinates for image coord: {obj['img_coord']}")
            obj['keypoint']= self.camera.get_world_coordinates(obj['img_coord'][0], obj['img_coord'][1])
            self.node.get_logger().info(f"Converted world coordinates: {obj['keypoint']}")
            if obj['keypoint'][0] == 0 and obj['keypoint'][1] == 0 and obj['keypoint'][2] == 0:
                obj['keypoint']= self.keypoints[idx]  
                self.node.get_logger().warning(f"Using previous keypoint due to conversion failure: {obj['keypoint']}")
            else:     
                self.keypoints[idx] = obj['keypoint']
            # NOTE: keypoint positions may be inaccurate due to noisy camera
            keypoint_positions.append(obj["keypoint"])
            if obj["is_grasped"]:
                print("grasping object")
                obj['keypoint']=self.get_ee_pos()
                self.keypoints[idx] = obj['keypoint']
        POINTS_CALLBACK=False
        return np.array(keypoint_positions)


    def get_object_by_keypoint(self, keypoint_idx):
        """
        Args:
            keypoint_idx (int): the index of the keypoint
        Returns:
            pointer: the object that the keypoint is associated with
        Given the keypoint index, this function returns the name of the object that the keypoint is associated with.
        """
        # assert hasattr(self, '_keypoint2object') and self._keypoint2object is not None, "Keypoints have not been registered yet."
        return self._keypoint_registry[keypoint_idx]["object"]


    def get_collision_points(self, noise=True):
        """Get collision points of gripper"""
        # Return mock collision points
        collision_points = np.random.rand(100, 3)
        return collision_points


    def reset(self):
        self.robot.set_to_home()

    def is_grasping(self, candidate_obj=None):
        
        # if the object is not graspable, then return False
        if candidate_obj == "none":
            return False
        else:
            for k in self._keypoint_registry:
                kp = self._keypoint_registry[k]
                if kp["object"] == candidate_obj:
                    # _, ee_point = self.endeffector.return_estimated_ee(self.camera, self.get_ee_pos())
                    # print("ee_point", ee_point)
                    # obj_point = kp["img_coord"]
                    ee_point=self.get_ee_pos()
                    obj_point=self.keypoints[k]
                    dist = np.linalg.norm(np.array(ee_point) - np.array(obj_point))
                    # Object is being grasped if the end effector is closed and close enough
                    # to the object keypoint # NOTE: This is not foolproof but it works
                    gripper_pos=self.robot.get_gripper_position()
                    if gripper_pos[1]>700:
                        gripper_state = int(0)
                    else:
                        gripper_state = int(1)
                    grasped = gripper_state & (dist < 40)
                    self._keypoint_registry[k]["is_grasped"] = grasped
                    self._keypoint_registry[k]["on_grasp_coord"] = ee_point
                    return grasped
                
    # from Euler 2 Quaternion
    def cart2quat(self,rot):
        """
        Convert Cartesian rotation (Euler angles) to quaternion.

        Args:
            rot (tuple or list): A tuple or list of three elements representing the rotation in Euler angles (roll, pitch, yaw).

        Returns:
            np.ndarray: A numpy array of four elements representing the quaternion (x, y, z, w).
        """
        r = R.from_euler('xyz', rot, degrees=True)
        quat = r.as_quat()
        return quat
    
    def quart2cart(self,quat):
        """
        Convert quaternion to Cartesian rotation (Euler angles).

        Args:
            quat (np.ndarray): A numpy array of four elements representing the quaternion (x, y, z, w).

        Returns:
            np.ndarray: A numpy array of three elements representing the rotation in Euler angles (roll, pitch, yaw).
        """
        r = R.from_quat(quat)
        rot = r.as_euler('xyz', degrees=True)
        return rot
        

    def get_ee_pose(self):
        position = np.array(self.robot.position[:3])  # 确保 position 是数组
        quat = self.get_ee_quat()
        return np.concatenate([position, quat])

    def get_ee_pos(self):
        return np.array(self.get_ee_pose()[:3])

    def get_ee_quat(self):
        euler=np.array(self.robot.position[3:])
        # print("euler:", euler)
        quat=self.cart2quat(euler)
        # print("quat:", quat)
        return quat
    
    def get_arm_joint_positions(self):
        return self.robot.get_servo_angle(is_radian=True)[1][:6] #or (is_radian=True)

    def set_ee_pose(self, pose):
        if isinstance(pose, np.ndarray):
            pose = pose.tolist()  # 转换为 Python 列表
        pose=np.concatenate([pose[:3], self.quart2cart(pose[3:])])
        # print("setting pose:", pose)
        self.robot.set_position(x=pose[0], y=pose[1], z=pose[2], roll=pose[3], pitch=pose[4], yaw=pose[5], speed=100, radius=10,is_radian=False,wait=True)  

    def close_gripper(self):
        self.robot.set_gripper_position(0)

    def open_gripper(self):
        self.robot.set_gripper_position(850)

    def get_last_og_gripper_action(self):
        return self.last_og_gripper_action
    
    def get_gripper_open_action(self):
        return -1.0
    
    def get_gripper_close_action(self):
        return 1.0
    
    def get_gripper_null_action(self):
        return 0.0
    
    def get_cam_obs(self):
        return self.camera.capture_image("rgb")
    
    
    def execute_action(
        self,
        action,
        precise=True,
    ):
        """
        Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

        Args:
            action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
            precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
        Returns:
            tuple: A tuple containing the position and rotation errors after reaching the target pose.
        """
    #     if precise:
    #         pos_threshold = 0.03
    #         rot_threshold = 3.0
    #     else:
    #         pos_threshold = 0.10
    #         rot_threshold = 5.0
    #     action = np.array(action).copy()
    #     assert action.shape == (8,)
    #     target_pose = action[:7]
    #     gripper_action = action[7]

    #     # # ======================================
    #     # # = status and safety check
    #     # # ======================================
    #     # if np.any(target_pose[:3] < self.bounds_min) \
    #     #         or np.any(target_pose[:3] > self.bounds_max):
    #     #     print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
    #     #     target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

    #     # # ======================================
    #     # # = interpolation
    #     # # ======================================
    #     # current_pose = self.get_ee_pose()
    #     # pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
    #     # rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
    #     # pos_is_close = pos_diff < self.interpolate_pos_step_size
    #     # rot_is_close = rot_diff < self.interpolate_rot_step_size
    #     # if pos_is_close and rot_is_close:
    #     #     self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
    #     #     pose_seq = np.array([target_pose])
    #     # else:
    #     #     num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
    #     #     pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
    #     #     self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')

    #     # ======================================
    #     # = move to target pose
    #     # ======================================
    #     # move faster for intermediate poses
    #     intermediate_pos_threshold = 0.10
    #     intermediate_rot_threshold = 5.0
    #     # for pose in target_pose:
    #         # self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
    #     self.set_ee_pose(action[:7])
    #     # move to the final pose with required precision
    #     # pose = pose_seq[-1]
    #     # # self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40) 
    #     # self.set_ee_pose(pose)
    #     # print("setting pose:", pose)
    #    # compute error
    #         # pos_error, rot_error = self.compute_target_delta_ee(target_pose)
    #     pos_error, rot_error = 0, 0
    #     self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

    #     # ======================================
    #     # = apply gripper action
    #     # ======================================
    #     if gripper_action == self.get_gripper_open_action():
    #         self.open_gripper()
    #     elif gripper_action == self.get_gripper_close_action():
    #         self.close_gripper()
    #     elif gripper_action == self.get_gripper_null_action():
    #         pass
    #     else:
    #         raise ValueError(f"Invalid gripper action: {gripper_action}")
        
    #     return pos_error, rot_error
        
        intermediate_pos_threshold = 0.10
        intermediate_rot_threshold = 5.0
        pose_seq = np.array(action)
        gripper_action = pose_seq[-1, -1]

        for pose in pose_seq[:-1]:
            # self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
            # print("setting pose:", pose[:3])
            self.set_ee_pose(pose[:7])
        # move to the final pose with required precision
        pose = pose_seq[-1]
        print("final pose:", pose[:3])       
        
        # self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40) 
        self.set_ee_pose(pose[:7])
         
        # compute error
        # pos_error, rot_error = self.compute_target_delta_ee(target_pose)
        pos_error, rot_error = 0, 0
        self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

        # ======================================
        # = apply gripper action
        # ======================================
        if gripper_action == self.get_gripper_open_action():
            self.open_gripper()
        elif gripper_action == self.get_gripper_close_action():
            self.close_gripper()
        elif gripper_action == self.get_gripper_null_action():
            pass
        else:
            raise ValueError(f"Invalid gripper action: {gripper_action}")
        return pos_error, rot_error

    def sleep(self, seconds):
        time.sleep(seconds)
    
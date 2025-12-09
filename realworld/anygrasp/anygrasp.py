import numpy as np
import torch
import open3d as o3d
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup, Grasp
import os
import imageio

# 配置参数
class Config:
    def __init__(self):
        self.checkpoint_path = "log/checkpoint_detection.tar"  # 模型路径
        self.max_gripper_width = 0.1
        self.gripper_height = 0.05
        self.top_down_grasp = False
        self.debug = True

cfgs = Config()

# 加载模型
anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()

def find_closest_grasp(gg, target_point):
    min_distance = 100
    closest_grasp = None
    print(len(gg))  
    for grasp in gg:
        grasp_position = grasp.translation
        distance = np.linalg.norm(grasp_position - target_point)
        if distance < min_distance:
            min_distance = distance
            closest_grasp = grasp
    print('min_distance:', min_distance)
    return closest_grasp
def world2cam_point(point_world, transform_matrix):
    T_mod = np.array([[1., 0., 0., 0., ],
                      [0., 1., 0., 0.,],
                      [0., 0., 1., 0.,],
                      [0., 0., 0., 1.,]])
    
    T_mod_inv = np.linalg.inv(T_mod)
    point_world_homogeneous = np.append(point_world, 1.0)
    point_cam_homogeneous = transform_matrix @ point_world_homogeneous
    point_cam_homogeneous = T_mod_inv @ point_cam_homogeneous
    point_cam = point_cam_homogeneous[:3] / point_cam_homogeneous[3]
    
    return point_cam

def cam2world_pose(position_cam, rotation_cam, transform_matrix):
    T_mod = np.array([[1., 0., 0., 0., ],
                      [0., 1., 0., 0.,],
                      [0., 0., 1., 0.,],
                      [0., 0., 0., 1.,]])
    
    T_mod_inv = np.linalg.inv(T_mod)
    position_cam_homogeneous = np.append(position_cam, 1.0)
    position_world_homogeneous = T_mod_inv @ position_cam_homogeneous
    position_world_homogeneous = transform_matrix @ position_world_homogeneous
    position_world = position_world_homogeneous[:3] / position_world_homogeneous[3]
    rotation_world = transform_matrix[:3, :3] @ T_mod_inv[:3, :3] @ rotation_cam
    R_link_from_opt = np.array([
        [0,  0,  1],
        [-1, 0,  0],
        [0, -1,  0]
    ], dtype=float)
    R_90_z = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ], dtype=float)
    rotation_world = rotation_world @ R_link_from_opt @ R_90_z
    return position_world, rotation_world


def main(rgb_path, depth_path, target_point_path, output_path):
    # Load images and target point
    rgb = imageio.imread(rgb_path)
    depth = imageio.imread(depth_path)
    target_point_world = np.load(target_point_path)
    print('target_point_world:', target_point_world)
    extrinsics = np.load('/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/camera_extrinsic1.npy')
    target_point = world2cam_point(target_point_world, np.linalg.inv(extrinsics))
    world_coordinate = np.array([0,0,0])
    world_coordinate_cam = world2cam_point(world_coordinate, np.linalg.inv(extrinsics))
    print('world_coordinate_cam:', world_coordinate_cam)
    print('target_point:', target_point)
    # Camera intrinsics
    scale = 1000.0
    intrinsics = np.load('/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/camera_intrinsic1.npy')
    fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
    print('fx, fy, cx, cy:', fx, fy, cx, cy)
    # Get point cloud
    xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / scale
    points_x = (xmap - cx) * points_z / fx 
    points_y = (ymap - cy) * points_z / fy 
    points = np.stack([points_x, points_y, points_z], axis=-1).reshape(-1, 3).astype(np.float32)
    colors = np.ascontiguousarray(rgb.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0)

    # Define workspace limits
    xmin, xmax = target_point[0] - 0.15, target_point[0] + 0.15
    ymin, ymax = target_point[1] - 0.15, target_point[1] + 0.15
    zmin, zmax = target_point[2] - 0.15, target_point[2] + 0.15
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # Find grasps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        anygrasp.net = anygrasp.net.cuda()

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if torch.cuda.is_available():
        anygrasp.net = anygrasp.net.cpu()
        torch.cuda.empty_cache()

    if gg is None or len(gg) == 0:
        print('No Grasp detected after collision detection!')
        np.save(output_path, None)
        return

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:60]
    print('grasp score:', gg_pick[0].score)
    trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    cloud.transform(trans_mat)
    grippers = gg_pick.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(trans_mat)
    target_point_cloud = o3d.geometry.PointCloud()
    target_point_cloud.points = o3d.utility.Vector3dVector([target_point])
    target_point_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
    target_point_cloud.transform(trans_mat)
    world_coordinate_cloud = o3d.geometry.PointCloud()
    world_coordinate_cloud.points = o3d.utility.Vector3dVector([world_coordinate_cam])
    world_coordinate_cloud.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # 绿色
    world_coordinate_cloud.transform(trans_mat)
    o3d.visualization.draw_geometries(grippers + [cloud, target_point_cloud, world_coordinate_cloud])
    
    #closest_grasp, _ = find_closest_grasp_weighted(gg, target_point)
    closest_grasp = find_closest_grasp(gg, target_point)
    print(closest_grasp)
    if closest_grasp is not None:
        grasp_position = closest_grasp.translation
        grasp_orientation = closest_grasp.rotation_matrix
        closest_gripper = closest_grasp.to_open3d_geometry()
        closest_gripper.transform(trans_mat)
        closest_gripper.paint_uniform_color([0.0, 1.0, 0.0]) 
        grasp_position_world, grasp_rotation_world = cam2world_pose(grasp_position, grasp_orientation, extrinsics)
        np.save(output_path, np.concatenate([grasp_position_world, grasp_rotation_world.flatten()]))
        o3d.visualization.draw_geometries([closest_gripper, cloud, world_coordinate_cloud])
        
    else:
        np.save(output_path, None)

if __name__ == '__main__':
    import sys
    rgb_save_path = "/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/anygrasp/rgb.png"
    depth_save_path = "/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/anygrasp/depth.png"
    target_point_path = "/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/anygrasp/target_point.npy"
    output_path = "/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/anygrasp/target_pose.npy"
    main(rgb_save_path, depth_save_path, target_point_path, output_path)

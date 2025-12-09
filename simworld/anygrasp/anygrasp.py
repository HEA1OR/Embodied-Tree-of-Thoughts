import numpy as np
import torch
import open3d as o3d
from PIL import Image
from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
import cv2

O3D_AXIS = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])

class Config:
    def __init__(self):
        self.checkpoint_path = "./log/checkpoint_detection.tar"  
        self.max_gripper_width = 0.1
        self.gripper_height = 0.04
        self.top_down_grasp = False
        self.debug = True

cfgs = Config()

anygrasp = AnyGrasp(cfgs)
anygrasp.load_net()

def cam2world_pose(position_cam, rotation_cam, transform_matrix):
    T_mod = np.array([[1., 0., 0., 0., ],
                      [0., -1., 0., 0.,],
                      [0., 0., -1., 0.,],
                      [0., 0., 0., 1.,]])
    T_mod_inv = np.linalg.inv(T_mod)
    position_cam_homogeneous = np.append(position_cam, 1.0)
    position_world_homogeneous = T_mod_inv @ position_cam_homogeneous
    position_world_homogeneous = transform_matrix @ position_world_homogeneous
    position_world = position_world_homogeneous[:3] / position_world_homogeneous[3]
    return position_world, rotation_cam

def world2cam_point(point_world, transform_matrix):
    T_mod = np.array([[1., 0., 0., 0., ],
                      [0., -1., 0., 0.,],
                      [0., 0., -1., 0.,],
                      [0., 0., 0., 1.,]])
    
    T_mod_inv = np.linalg.inv(T_mod)
    point_world_homogeneous = np.append(point_world, 1.0)
    point_cam_homogeneous = transform_matrix @ point_world_homogeneous
    point_cam_homogeneous = T_mod_inv @ point_cam_homogeneous
    point_cam = point_cam_homogeneous[:3] / point_cam_homogeneous[3]
    return point_cam

def cam2world_rotation(rotation_cam, transform_matrix):
    T_mod = np.array([[1., 0., 0., 0., ],
                      [0., -1., 0., 0.,],
                      [0., 0., -1., 0.,],
                      [0., 0., 0., 1.,]])
    T_mod_inv = np.linalg.inv(T_mod)
    rotation_world = transform_matrix[:3, :3] @ T_mod_inv[:3, :3] @ rotation_cam
    return rotation_world


def read_images(color_path, depth_path):
    color_image = np.array(Image.open(color_path)) / 255.0
    depth_image = np.array(Image.open(depth_path))
    return color_image, depth_image

def generate_point_cloud(depth_image, fx, fy, cx, cy, scale):
    xmap, ymap = np.arange(depth_image.shape[1]), np.arange(depth_image.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth_image / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z
    points = np.stack([points_x, points_y, points_z], axis=-1)
    return points


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

def generate_grasp(color_path, depth_path, intrinsics, extrinsics, target_point_world):
    color_image, depth_image = read_images(color_path, depth_path)
    fx, fy, cx, cy = intrinsics[0][0], intrinsics[1][1], intrinsics[0][2], intrinsics[1][2]
    scale = 1000.0
    points = generate_point_cloud(depth_image, fx, fy, cx, cy, scale)
    mask = (points[:, :, 2] > 0) & (points[:, :, 2] < 2)
    points = points[mask].astype(np.float32)
    colors = color_image[mask].astype(np.float32)
    
    target_point_cam = world2cam_point(target_point_world, extrinsics)
    print('target point cam:', target_point_cam)
    xmin, xmax = target_point_cam[0] - 0.1, target_point_cam[0] + 0.1
    ymin, ymax = target_point_cam[1] - 0.1, target_point_cam[1] + 0.1
    zmin, zmax = target_point_cam[2] - 0.1, target_point_cam[2] + 0.1
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    
    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
    
    print(len(gg))
    if len(gg) == 0:
        print('No Grasp detected after collision detection!')
        return None

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]

    trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    cloud.transform(trans_mat)

    grippers = gg_pick.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(trans_mat)
    target_point_cam = world2cam_point(target_point_world, extrinsics)
    print('target point cam:', target_point_cam)
    closest_grasp = find_closest_grasp(gg_pick, target_point_cam)
    
    if closest_grasp is not None:
        print('closest grasp:', closest_grasp)
        closest_gripper = closest_grasp.to_open3d_geometry()
        closest_gripper.transform(trans_mat)
        target_point_cloud = o3d.geometry.PointCloud()
        target_point_cloud.points = o3d.utility.Vector3dVector([target_point_cam])
        target_point_cloud.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色
        target_point_cloud.transform(trans_mat)
        #o3d.visualization.draw_geometries([closest_gripper, cloud, target_point_cloud, O3D_AXIS])
        grasp_position = closest_grasp.translation
        grasp_orientation = closest_grasp.rotation_matrix
        width = closest_grasp.width
        print(f"Closest Grasp position: {grasp_position}")
        print(f"Grasp rotation matrix: {grasp_orientation}")
        grasp_position_world, grasp_rotation_world = cam2world_pose(grasp_position, grasp_orientation, extrinsics_inv)
        return grasp_position_world, grasp_rotation_world, width
    else:
        print("No closest grasp found.")
        #o3d.visualization.draw_geometries([cloud, O3D_AXIS])



# 示例调用
if __name__ == "__main__":
    color_path = "rgb.png"
    depth_path = "depth.png"
    intrinsics = np.load('intrinsics.npy')
    extrinsics = np.load('extrinsics.npy')
    extrinsics_inv = np.linalg.inv(extrinsics)
    target_point_world = np.load('target_point.npy')
    print('target point world:', target_point_world)
    grasp_pose_world = generate_grasp(color_path, depth_path, intrinsics, extrinsics, target_point_world)
    if grasp_pose_world is not None:
        grasp_position, grasp_orientation, width = grasp_pose_world
        print("Grasp Pose:")
        print("Position:", grasp_position)
        print("Orientation:", grasp_orientation)
        print('width:', width)
        grasp_orientation_mat = cam2world_rotation(grasp_orientation, extrinsics_inv)
        print(grasp_orientation_mat)
        from scipy.spatial.transform import Rotation as R
        q = R.from_matrix(grasp_orientation_mat).as_quat()
        print('quaternion:', q)
        np.save('target_pose.npy', np.concatenate([grasp_position, q]))
        np.save('width.npy', width)
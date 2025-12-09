#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

# hk-v9plh01000058467e8a9fb20e9f65263364d543cda064659

# # Your camera extrinsics (link_base -> camera)
# # translation = np.array([0.863182, 0.107854, 0.219297])
# # translation = np.array([-0.095368, -0.468431, 0.397804])
# translation = np.array([-0.095368, -0.408431, 0.457804])
# quaternion = np.array([-0.004694, 0.267570, 0.192732, 0.944054])  # [x, y, z, w]
# # translation = np.array([0.9299, 0.1041, 0.1973])
# # quaternion = np.array([0.123981, -0.093213, -0.958927, 0.237483])  # [x, y, z, w]

# # Convert quaternion to rotation matrix
# rotation_matrix = R.from_quat(quaternion).as_matrix()

# # Create 4x4 SE(3) transformation matrix
# T_base_camera_standard = np.eye(4)
# T_base_camera_standard[:3, :3] = rotation_matrix
# T_base_camera_standard[:3, 3] = translation

# print("SE(3) Transformation Matrix (link_base -> camera):")
# print(T_base_camera_standard)
# print()

# # RealSense camera coordinate frame transformation
# # Standard robotics: forward=+x, right=+y, up=+z
# # RealSense camera: forward=+z, right=+x, down=+y
# # Transformation matrix from standard to RealSense frame:
# T_standard_to_realsense = np.array([
#     [0,  0,  1,  0],  # RealSense +x = Standard +z
#     [0, -1,  0,  0],  # RealSense +y = Standard -y (down vs up)
#     [1,  0,  0,  0],  # RealSense +z = Standard +x
#     [0,  0,  0,  1]
# ])

# # Apply coordinate frame transformation
# T_base_camera_realsense = T_base_camera_standard @ T_standard_to_realsense

# print("SE(3) Transformation Matrix (link_base -> camera_realsense):")
# print(T_base_camera_realsense)
# print()

# # take inverse of T_base_camera to get T_camera_base
# T_camera_base = np.linalg.inv(T_base_camera_realsense)

# print("SE(3) Transformation Matrix (camera -> link_base):")
# print(T_camera_base)
# print()

t_bc = np.array([0.72, 0.07, 0.62])  # meters
q_bc = np.array([0.528302, -0.017558, -0.844618, 0.084912])  # [x,y,z,w]
R_bc = R.from_quat(q_bc).as_matrix()
K = np.array([
            [608, 0, 642],
            [0, 608, 363.6],
            [0, 0, 1]
        ])
# Fixed: camera_link <- color_optical_frame (optical -> link)
R_link_from_opt = np.array([
    [0,  0,  1],
    [-1, 0,  0],
    [0, -1,  0]
], dtype=float)

# Compose base <- optical
R_base_from_opt = R_bc @ R_link_from_opt
T_base_from_opt = np.eye(4)
T_base_from_opt[:3,:3] = R_base_from_opt
T_base_from_opt[:3, 3]  = t_bc

# Save as camera_extrinsic1.npy as specified in README
np.save("camera_extrinsic1.npy", T_base_from_opt)
print("Saved camera extrinsics to camera_extrinsic1.npy")
# Save as camera_intrinsic1.npy as specified in README
np.save("camera_intrinsic1.npy", K)
print(K)

# Verify the saved file
loaded_extrinsics = np.load("camera_extrinsic1.npy")
print("\nVerification - Loaded matrix:")
print(loaded_extrinsics)


p_opt = np.array([-0.0975, -0.0515, 1.03])
# p_opt *= 1000.0  # convert to mm
# add 1 for homogeneous coordinates
p_opt = np.hstack((p_opt, 1.0))  # shape (==4,)

print("Optical points: ", p_opt[:3])
print("Base from optical T", (T_base_from_opt @ p_opt)[:3])

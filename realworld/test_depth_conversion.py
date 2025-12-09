#!/usr/bin/env python3

"""
Test script to diagnose depth-to-coordinate conversion issues
"""

import rclpy
from realsense_camera_ros import RealSenseCamera
import time
import numpy as np

def test_depth_conversion():
    """Test the depth conversion with debug output"""
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Create camera instance
        camera = RealSenseCamera()
        
        print("Waiting for camera data to be available...")
        
        # Wait for data to be available
        timeout = 10.0  # 10 seconds timeout
        start_time = time.time()
        
        while not (camera.received_rgb_image and camera.received_depth_image and camera.received_camera_info):
            rclpy.spin_once(camera, timeout_sec=0.1)
            if time.time() - start_time > timeout:
                print("Timeout waiting for camera data!")
                return
                
        print("Camera data is available!")
        
        # Test pixel coordinates - use center of image first
        test_coordinates = [
            (640, 360),  # Center for 1280x720
            (320, 240),  # Center for 640x480
            (998, 508),  # The problematic coordinate from error message
            (100, 100),  # Top-left area
            (1000, 500), # Another test coordinate
        ]
        
        for x, y in test_coordinates:
            print(f"\n=== Testing coordinates ({x}, {y}) ===")
            
            try:
                # Get RGB and depth images to check dimensions
                rgb_image = camera.capture_image("rgb")
                depth_image = camera.capture_image("depth")
                
                print(f"RGB image shape: {rgb_image.shape}")
                print(f"Depth image shape: {depth_image.shape}")
                
                # Check if coordinates are within bounds
                if not (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
                    print(f"Coordinates ({x}, {y}) are outside depth image bounds!")
                    continue
                    
                print("converting to world coordinates...")  # This matches the error message!
                
                # Test depth conversion
                world_coords = camera.get_world_coordinates(x, y)
                print(f"World coordinates: {world_coords}")
                
            except Exception as e:
                print(f"Error converting coordinates ({x}, {y}): {e}")
                
        # Test a small region to see depth statistics
        print(f"\n=== Depth Statistics Analysis ===")
        depth_image = camera.capture_image("depth")
        
        # Sample a region around the center
        center_x, center_y = depth_image.shape[1] // 2, depth_image.shape[0] // 2
        region = depth_image[center_y-50:center_y+50, center_x-50:center_x+50]
        
        print(f"Center region ({center_x-50}:{center_x+50}, {center_y-50}:{center_y+50}) statistics:")
        print(f"  Min depth: {np.min(region)}")
        print(f"  Max depth: {np.max(region)}")
        print(f"  Mean depth: {np.mean(region)}")
        print(f"  Non-zero pixels: {np.count_nonzero(region)}/{region.size}")
        print(f"  Zero pixels: {np.count_nonzero(region == 0)}")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
    except Exception as e:
        print(f"Test failed with error: {e}")
    finally:
        # Cleanup
        if 'camera' in locals():
            camera.close()
        rclpy.shutdown()

import numpy as np
from scipy.spatial.transform import Rotation as R

# TF says: parent=link_base, child=camera_link
t_bc = np.array([-0.095370, -0.408400, 0.457804])  # meters
q_bc = np.array([-0.004694, 0.267570, 0.192732, 0.944054])  # [x,y,z,w]
R_bc = R.from_quat(q_bc).as_matrix()

def camera_to_base(p_c):
    p_c = np.asarray(p_c).reshape(3,)
    return R_bc @ p_c + t_bc  # <-- base ← camera

translation = np.array([-0.095368, -0.408431, 0.457804])
quaternion = np.array([-0.004694, 0.267570, 0.192732, 0.944054])  # [x, y, z, w]
# translation = np.array([0.9299, 0.1041, 0.1973])
# quaternion = np.array([0.123981, -0.093213, -0.958927, 0.237483])  # [x, y, z, w]

# Convert quaternion to rotation matrix
rotation_matrix = R.from_quat(quaternion).as_matrix()

# Create 4x4 SE(3) transformation matrix
T_base_camera_standard = np.eye(4)
T_base_camera_standard[:3, :3] = rotation_matrix
T_base_camera_standard[:3, 3] = translation

print("SE(3) Transformation Matrix (link_base -> camera):")
print(T_base_camera_standard)
print()

# Fixed optical -> camera_link rotation
R_link_from_opt = np.array([
    [0,  0,  1],
    [-1, 0,  0],
    [0, -1,  0],
], dtype=float)

def optical_to_base(p_optical_xyz_m, t_bc_m, q_bc_xyzw):
    """
    Convert a 3D point from color_*_optical_frame to base_link.

    Args:
        p_optical_xyz_m : (3,) array-like, point in optical frame [meters]
        t_bc_m          : (3,) array-like, translation of base->camera_link [meters]
        q_bc_xyzw       : (4,) array-like, quaternion [x,y,z,w] for base->camera_link

    Returns:
        (3,) ndarray, point in base_link [meters]
    """
    p_opt = np.asarray(p_optical_xyz_m, dtype=float).reshape(3,)
    t_bc  = np.asarray(t_bc_m,          dtype=float).reshape(3,)
    R_bc  = R.from_quat(np.asarray(q_bc_xyzw, dtype=float)).as_matrix()

    # optical -> camera_link
    p_link = R_link_from_opt @ p_opt
    # camera_link -> base (using base <- camera transform)
    p_base = R_bc @ p_link + t_bc
    return p_base

t_bc = np.array([-0.095370, -0.408400, 0.457804])  # meters
q_bc = np.array([-0.004694, 0.267570, 0.192732, 0.944054])  # [x,y,z,w]
R_bc = R.from_quat(q_bc).as_matrix()

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

if __name__ == "__main__":
    # Example: point in camera frame (meters)
    # p_c = np.array([-0.117338, -0.065865, 1.085111])
    # p_c = np.array([1.03, 0.0975, 0.0515])
    p_c = np.array([-0.0975, -0.0515, 1.03])
    # p_c = np.array([0.709, 0.0928, -0.139])
    # p_b = camera_to_base(p_c)
    p_b = optical_to_base(p_c, t_bc, q_bc)

    print("Camera point:", p_c)
    print("Base point:", p_b)

    p_opt = np.array([p_c[0], p_c[1], p_c[2], 1])

    print("Base from optical T", T_base_from_opt @ p_opt)
    # test_depth_conversion()

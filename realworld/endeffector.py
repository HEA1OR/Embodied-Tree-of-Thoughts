from xarm.wrapper import XArmAPI  # 引入 xArm SDK
from realsense_camera_ros import RealSenseCamera
import cv2
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from skimage.draw import disk

class EndEffector:

    def __init__(self):
        print("EndEffector")
        self.LLIM = [[100,700], [-400, +400], [100, 600]]  # x, y, z limits in mm

    def get_point_to_world_conversion(self, camera):
        
        # Get limits of end-effector
        x_limits, y_limits, z_limits = self.LLIM[0], self.LLIM[1], self.LLIM[2]
        x_interp = np.linspace(x_limits[0], x_limits[1], num=100)
        y_interp = np.linspace(y_limits[0], y_limits[1], num=100)
        z_interp = np.linspace(z_limits[0], z_limits[1], num=100)

        # Iterate through points
        print("Getting point to world dict...")
        t = time.time()
        self.point_to_world = {}
        for x in x_interp:
            for y in y_interp:
                for z in z_interp:
                    point = self.convert_world_to_point(camera, [x, y, z])
                    self.point_to_world[tuple(point)] = [x,y,z]
        print("Time:", time.time() - t)
        print("point_to_world initialized successfully.")
        return self.point_to_world

    def convert_world_to_point(self,cam_node, world_coord):
        T = np.hstack((cam_node.R, cam_node.t))
        P = np.array([[world_coord[0], world_coord[1], world_coord[2], 1]])
        pc = T @ P.T
        x_star = cam_node.K @ pc

        # 添加保护以防止除以零
        if abs(x_star[2]) < 1e-10:  # 添加一个小的阈值
            print(f"警告：投影计算中遇到接近零的值: {x_star[2]}")
            return [0, 0]  # 或者返回一个默认值或者抛出特定的异常

        x_star = x_star / x_star[2]
        point = [int(np.rint(x_star[1])[0]), int(np.rint(x_star[0])[0])]
        # print(f"[EndEffector] Converted world coord {world_coord} to pixel point {point}")
        return point

    def return_estimated_ee(self, cam_node, curr_position):
        """
        Get estimated pixel coordinate position of the end-effector.
        """

        # Compute pixel coordinate
        new_world = curr_position   
        point = self.convert_world_to_point(cam_node, new_world)
        rgb = cam_node.capture_image("rgb")
        rr, cc = disk(point, 10, shape=rgb.shape)
        rgb[rr, cc] = (255, 255, 0)

        return rgb, point
    
    def find_closest_point_to_world(self, ref_point):

        # Convert dictionary keys (pixel points) to a NumPy array for fast distance computation
        pixel_points = np.array(list(self.point_to_world.keys()))
        
        # Compute Euclidean distances from ref_point to all pixel points
        distances = np.linalg.norm(pixel_points - np.array(ref_point), axis=1)
        
        # Find the index of the closest pixel point
        closest_index = np.argmin(distances)
        
        # Retrieve the closest pixel point and its corresponding world point
        closest_pixel_point = tuple(pixel_points[closest_index])  # Convert back to tuple
        corresponding_world_point = self.point_to_world[closest_pixel_point]

        print(f"[EndEffector] Closest pixel point to {ref_point} is {closest_pixel_point} with world coord {corresponding_world_point}")

        return corresponding_world_point


    def exit(self):
        """
        Safely disconnects the robot.
        """
        input("Press return to deactivate robot...")
        self.disconnect()
        print("xArm Disconnected!")


if __name__ == "__main__":

    PORT = "/dev/ttyACM0"
    ENABLE_TORQUE = True

    koch_robot = KochRobot(port=PORT, torque=ENABLE_TORQUE)
    cam_node = RealSenseCamera()

    try:
        do = input("Action (read=r, manual=m, calib=c):")
        if do == "r":
            print(koch_robot.get_ee_pose())
        elif do == "m":
            koch_robot.manual_control(cam_node)
        elif do == "c":
            # pass
            koch_robot.camera_extrinsics(cam_node)
    except KeyboardInterrupt:
        koch_robot.exit()

    koch_robot.exit()
    

    

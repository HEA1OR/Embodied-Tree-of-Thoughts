"""
Adapted from OmniGibson and the Lula IK solver for XArm.
This code provides an inverse kinematics (IK) solver for the XArm robot using the XArmAPI.
"""

import numpy as np

class IKResult:
    """Class to store IK solution results"""
    def __init__(self, success, joint_positions, error_pos, error_rot, num_descents=None):
        self.success = success
        self.cspace_position = joint_positions
        self.position_error = error_pos
        self.rotation_error = error_rot
        self.num_descents = num_descents if num_descents is not None else 1


class xArmIKSolver:

    def __init__(self, arm):
        """
        Initialize with XArmAPI instance.
        :param arm: The XArmAPI object for controlling the arm.
        """
        self.arm = arm

    def solve(self, target_pose_homo,
              position_tolerance=0.01,
              orientation_tolerance=0.05,
              max_iterations=150,
              initial_joint_pos=None):
        """
        IK solver for XArm robot.
        """

        # Extract position and orientation from homogeneous transformation matrix
        target_pos = target_pose_homo[:3, 3]  # The translation part (x, y, z)
        target_rot = target_pose_homo[:3, :3]  # The rotation part (3x3 matrix)

        # Create pose as [x, y, z, roll, pitch, yaw] for XArm
        # Convert the rotation matrix to roll, pitch, yaw (Euler angles)
        # Assuming you have a function `rotation_matrix_to_euler_angles` that converts rotation matrix to roll, pitch, yaw
        roll, pitch, yaw = self.rotation_matrix_to_euler_angles(target_rot)
        pose = target_pos.tolist() + [roll, pitch, yaw]

        # Use the XArm API to compute the inverse kinematics
        result_code, joint_angles = self.arm.get_inverse_kinematics(pose, input_is_radian=True, return_is_radian=True)
        # print("result_code",result_code)
        if result_code == 0:  # Success
            return IKResult(success=True,
                            joint_positions=np.array(joint_angles[:6]),
                            error_pos=0,  # You can calculate this if you need
                            error_rot=0,  # Similarly, compute rotation error if needed
                            num_descents=1)
        else:
            # If the inverse kinematics fails, return the initial joint positions as a fallback
            # if initial_joint_pos is not None:
            #     # 确保 initial_joint_pos 是平坦的一维数组
            #     initial_joint_pos = np.concatenate([np.array(joint) for joint in initial_joint_pos]) if isinstance(initial_joint_pos[0], (list, np.ndarray)) else np.array(initial_joint_pos)
            # else:
            #     initial_joint_pos = None
            return IKResult(success=False,
                            joint_positions = initial_joint_pos,
                            error_pos=0,
                            error_rot=0,
                            num_descents=1)

    def rotation_matrix_to_euler_angles(self, R):
        """
        Convert a rotation matrix to Euler angles (roll, pitch, yaw)
        :param R: 3x3 rotation matrix
        :return: roll, pitch, yaw (in radians)
        """
        # Assuming R is a 3x3 rotation matrix
        pitch = np.arctan2(R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        roll = np.arctan2(R[1, 0], R[0, 0])
        yaw = np.arctan2(R[2, 1], R[2, 2])
        return roll, pitch, yaw

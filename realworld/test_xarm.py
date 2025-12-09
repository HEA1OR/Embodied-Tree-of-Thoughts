from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation as R
def cart2quat(rot):
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

ip = '192.168.1.225'

robot = XArmAPI(ip)
robot.motion_enable(True)
robot.clean_error()
robot.set_gripper_mode(0)
robot.set_gripper_enable(True)
robot.set_gripper_position(850, wait=True)
_ = input("Press Enter to continue...")
robot.set_mode(0)
robot.set_state(0)
robot.set_position(300, 0, 0, 180, 0, 0, wait=True)
# robot.set_position(300, 0, 0, 180, 0, 30, wait=True)
pos = robot.get_position()[-1]
print(pos)
print(cart2quat((pos[3], pos[4], pos[5])))
_ = input("Press Enter to continue...")
robot.set_position(500, 0, 0, 180, 0, 0, wait=True)
robot.set_position(500, 0, 0, 210, 0, 0, wait=True)
robot.set_position(500, 0, 0, 150, 0, 0, wait=True)
robot.set_position(500, 0, 0, 180, 0, 0, wait=True)

_ = input("Press Enter to continue...")
robot.set_position(500, 0, 0, 180, -90, 0, wait=True)
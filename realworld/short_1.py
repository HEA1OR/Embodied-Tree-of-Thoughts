import os, cv2, numpy as np, time, rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from geometry_msgs.msg import Point
from xarm.wrapper import XArmAPI
from realsense_camera_ros import RealSenseCamera
from endeffector import EndEffector
import torch
from scipy.spatial.transform import Rotation as R
import rekep.transform_utils as T
from scipy.optimize import minimize

# ---------------- Parameters ----------------
MASK_DIR      = "mask"
Z_APPROACH    = 205    # pre-grasp height (mm)
Z_GRASP       = 40     # grasping height (mm)
Z_LIFT        = 150
GRIPPER_OPEN  = 850
GRIPPER_CLOSE = 150
GRASP_TOPIC   = "/grasp_pose"
# -------------------------------------------------------------

def rotate_around_y(quat, theta):
    rot_mat = R.from_quat(quat).as_matrix()
    Ry = np.array([
        [np.cos(theta[0]), 0, np.sin(theta[0])],
        [0, 1, 0],
        [-np.sin(theta[0]), 0, np.cos(theta[0])]
    ])
    return R.from_matrix(rot_mat @ Ry).as_quat()

def rotate_around_z(quat, theta):
    rot_mat = R.from_quat(quat).as_matrix()
    Rz = np.array([
        [1, 0, 0],
        [0, np.cos(theta[0]), -np.sin(theta[0])],
        [0, np.sin(theta[0]),  np.cos(theta[0])]
    ])
    return R.from_matrix(rot_mat @ Rz).as_quat()

def objective_y(theta, ori, target_x):
    rotated = R.from_quat(rotate_around_y(ori, theta)).apply([0, 0, 1])
    return np.arccos(np.clip(np.dot(rotated, target_x), -1.0, 1.0))

def objective_z(theta, ori, target_z):
    rotated = R.from_quat(rotate_around_z(ori, theta)).apply([0, 0, 1])
    return np.arccos(np.clip(np.dot(rotated, target_z), -1.0, 1.0))

def optimize_y(ori, target_x):
    res = minimize(objective_y, 0.0, args=(ori, target_x), bounds=[(-np.pi, np.pi)])
    return rotate_around_y(ori, res.x)

def optimize_z(ori, target_z):
    res = minimize(objective_z, 0.1, args=(ori, target_z), bounds=[(-np.pi, np.pi)])
    return rotate_around_z(ori, res.x)


# ==============================================================
#                       Main ROS Node
# ==============================================================
class MinimalRekepNode(Node):
    def __init__(self):
        super().__init__('minimal_rekep')

        # ---------- Robot Initialization ----------
        ip = '192.168.1.225'
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(True)
        self.arm.clean_error()
        self.arm.set_mode(6)
        self.arm.set_state(0)
        self.arm.set_servo_angle(angle=[0,-33,-57,0,90,0], speed=50)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_position(GRIPPER_OPEN, wait=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.get_logger().info("xArm initialized.")
        time.sleep(1)

        # ---------- Seeds ----------
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # ---------- Camera ----------
        self.cam = RealSenseCamera()

        # ---------- ROS Interfaces ----------
        self.keypoints = None
        self.latest_grasp = None
        self.target_pub = self.create_publisher(Point, '/target_point', 10)
        self.grasp_sub  = self.create_subscription(
            Float32MultiArray, GRASP_TOPIC, self._grasp_cb, 10)

    # ----------------------------------------------------------
    #   Keypoint selection (user clicks in the image)
    # ----------------------------------------------------------
    def _load_keypoints_once(self):
        rgb = self.cam.capture_image("rgb")
        points = self.cam.pixel_to_3d_points()

        k3d, k2d, proj = self.get_keypoints(rgb, points, max_keypoints=8)
        self.get_logger().info(f'Selected {len(k3d)} keypoints.')

        for i, p in enumerate(k2d):
            coord = self.cam.get_world_coordinates(p[1], p[0])
            k3d[i] = coord
            self.get_logger().info(f"Pixel {p} -> World {coord}")

        cv2.imwrite(os.path.join(MASK_DIR, "projected.png"), proj)
        return k3d

    def project_keypoints_to_img(self, rgb, pixels):
        img = rgb.copy()
        for i, px in enumerate(pixels):
            cv2.rectangle(img, (px[1]-15, px[0]-15), (px[1]+15, px[0]+15), (255,255,255), -1)
            cv2.rectangle(img, (px[1]-15, px[0]-15), (px[1]+15, px[0]+15), (0,0,0), 2)
            cv2.putText(img, str(i), (px[1]-10, px[0]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        return img

    def get_keypoints(self, rgb, points, max_keypoints=8):
        assert rgb.shape[:2] == points.shape[:2]
        points = np.nan_to_num(points)

        img_disp = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).copy()
        pts2d, pts3d = [], []
        win = 'KeypointSelector'

        def mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts2d) < max_keypoints:
                z = points[y, x, 2]
                if np.isfinite(z):
                    pts2d.append((y, x))
                    pts3d.append((y, x, z))
                    cv2.circle(img_disp, (x, y), 4, (0,0,255), -1)
                    cv2.imshow(win, img_disp)

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, mouse)
        cv2.imshow(win, img_disp)
        print("Left-click to select keypoints; press 'q' to finish.")

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyWindow(win)

        pixels = np.array(pts2d, dtype=np.int32)
        kpts   = np.array(pts3d, dtype=np.float32)
        proj   = self.project_keypoints_to_img(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB), pixels)
        return kpts, pixels, proj

    # ----------------------------------------------------------
    #   Grasp pose callbacks
    # ----------------------------------------------------------
    def _grasp_cb(self, msg):
        self.latest_grasp = np.array(msg.data)
        self.get_logger().info(f'Received grasp pose: {self.latest_grasp[:3]}')

    def _wait_grasp(self, timeout=100.0):
        t0 = time.time()
        while self.latest_grasp is None and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        g = self.latest_grasp
        self.latest_grasp = None
        return g

    def _request_grasp(self, kp_id):
        pt = Point()
        xyz = self.keypoints[kp_id] / 1000.0
        pt.x, pt.y, pt.z = xyz
        pt.z += 0.08
        self.target_pub.publish(pt)
        self.get_logger().info(f'Sent candidate point kp{kp_id}')

    # ----------------------------------------------------------
    #   Pose conversion and movement utilities
    # ----------------------------------------------------------
    def quart2cart(self, quat):
        return R.from_quat(quat).as_euler('xyz', degrees=True)

    def set_ee_pose(self, pose):
        pose = np.concatenate([pose[:3], self.quart2cart(pose[3:])])
        self.arm.set_position(
            x=pose[0], y=pose[1], z=pose[2],
            roll=pose[3], pitch=pose[4], yaw=pose[5],
            speed=60, radius=10, is_radian=False, wait=True
        )

    def offset_pose(self, pose, dz):
        homo = T.convert_pose_quat2mat(pose)
        pose[:3] += homo[:3, :3] @ np.array([0, 0, dz])
        return pose

    # ----------------------------------------------------------
    #   Pick & Place primitives
    # ----------------------------------------------------------
    def pick(self, kp_id, optimize=False):
        self.get_logger().info(f'>>> PICK kp{kp_id}')
        self._request_grasp(kp_id)
        grasp12 = self._wait_grasp()

        if grasp12 is None:
            pose = np.r_[self.keypoints[kp_id] + [0, 0, Z_APPROACH], [1,0,0,0]]
        else:
            pos = grasp12[:3] * 1000
            quat = R.from_matrix(grasp12[3:].reshape(3,3)).as_quat()
            pose = np.r_[pos, quat]

        if optimize:
            target = np.array([0, 0, -1])
            ori = optimize_z(optimize_y(quat, target), target)
        else:
            ori = quat

        pose[3:] = ori
        pose = self.offset_pose(pose, -Z_APPROACH)
        time.sleep(2.0)
        self._move(pose)

        pose = self.offset_pose(pose, Z_GRASP)
        self._move(pose)

        self._move(pose, GRIPPER_CLOSE)
        pose = self.offset_pose(pose, -Z_LIFT)
        self._move(pose)

    def place(self, kp_id):
        self.get_logger().info(f'>>> PLACE kp{kp_id}')
        pose = np.r_[self.keypoints[kp_id], [1,0,0,0]]
        pose = self.offset_pose(pose, -1.2 * Z_APPROACH)
        self._move(pose, GRIPPER_CLOSE)
        self._move(pose, GRIPPER_OPEN)

    def _move(self, pose, gripper=None, wait=True):
        self.set_ee_pose(pose)
        if gripper is not None:
            self.arm.set_gripper_position(gripper, wait=True)

    # ----------------------------------------------------------
    #   Example task: pick keypoint 0 and place to keypoint 1
    # ----------------------------------------------------------
    def run_task(self):
        while self.cam.rgb_image is None or self.cam.depth_image is None:
            self.get_logger().info("Waiting for camera frames...")
            time.sleep(0.5)

        self.keypoints = self._load_keypoints_once()
        self.get_logger().info(f'Loaded {len(self.keypoints)} keypoints')

        self.pick(0, optimize=False)
        self.place(1)

    def shutdown(self):
        self.arm.set_servo_angle([0,-33,-57,0,90,0], wait=True)
        self.arm.disconnect()
        self.cam.close()


# ==============================================================
#                            Main entry
# ==============================================================
def main():
    rclpy.init()

    from rclpy.executors import MultiThreadedExecutor
    import threading

    main_node = MinimalRekepNode()
    cam_node  = main_node.cam

    executor = MultiThreadedExecutor()
    executor.add_node(main_node)
    executor.add_node(cam_node)

    def task():
        main_node.run_task()

    t = threading.Thread(target=task)
    t.start()

    while t.is_alive():
        executor.spin_once(timeout_sec=0.1)

    main_node.destroy_node()
    cam_node.destroy_node()
    executor.shutdown()
    rclpy.shutdown()
    t.join()


if __name__ == '__main__':
    main()

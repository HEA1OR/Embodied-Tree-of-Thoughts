import os
import subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
import imageio

class AnyGraspNode(Node):
    def __init__(self):
        super().__init__('grasp_detection_node')
        
        # 初始化 CvBridge
        self.bridge = CvBridge()
        
        # 创建发布者
        self.grasp_pub = self.create_publisher(Float32MultiArray, '/grasp_pose', 10)
        
        # 创建订阅者
        self.color_sub = self.create_subscription(ROSImage, '/rgb/image_raw', self.color_callback, 10)
        self.depth_sub = self.create_subscription(ROSImage, '/depth_to_rgb/image_raw', self.depth_callback, 10)
        self.target_sub = self.create_subscription(Point, '/target_point', self.grasp_callback, 10)
        
        self.received_color_image = False
        self.received_depth_image = False
        self.color_image = None
        self.depth_image = None

        self.get_logger().info('AnyGraspNode initialized')

    def color_callback(self, msg):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert color image: {e}")
        if not self.received_color_image:
            self.get_logger().info("Received first color image.")
            self.received_color_image = True

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")
        if not self.received_depth_image:
            self.get_logger().info("Received first depth image.")
            self.received_depth_image = True

    def grasp_callback(self, msg):
        if self.color_image is None or self.depth_image is None:
            self.get_logger().warn("Color or depth image is not available.")
            return

        target_point = np.array([msg.x, msg.y, msg.z])

        rgb_save_path = "/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/anygrasp/rgb.png"
        depth_save_path = "/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/anygrasp/depth.png"
        target_point_path = "/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/anygrasp/target_point.npy"
        output_path = "/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/anygrasp/target_pose.npy"

        imageio.imwrite(rgb_save_path, self.color_image)
        imageio.imwrite(depth_save_path, self.depth_image)
        np.save(target_point_path, target_point)

        result = subprocess.run(
            ["conda", "run", "-n", "anygrasp", "python", "./anygrasp.py", rgb_save_path,depth_save_path, target_point_path, output_path],
capture_output=True,
text=True
)
        if result.returncode != 0:
            self.get_logger().error(f"Failed to run grasp detection script: {result.stderr}")
            return

        # 读取抓取位姿
        target_pose = np.load(output_path)

        if target_pose is None:
            self.get_logger().info("No grasp detected.")
            return

        grasp_position = target_pose[:3]
        grasp_orientation = target_pose[3:].reshape(3, 3)

        # 发布抓取位姿
        result_msg = Float32MultiArray()
        result_msg.data = grasp_position.tolist() + grasp_orientation.flatten().tolist()
        self.grasp_pub.publish(result_msg)
        self.get_logger().info(f"Grasp pose sent: {grasp_position}")
        self.get_logger().info(f"Grasp orientation sent: {grasp_orientation}")

def main(args=None):
    rclpy.init(args=args)
    try:
        node = AnyGraspNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
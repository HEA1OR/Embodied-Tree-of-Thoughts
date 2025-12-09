import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class UndistortNode(Node):
    def __init__(self):
        super().__init__('undistort_node')
        self.bridge = CvBridge()
        
        # 订阅原始图像话题
        self.rgb_sub = self.create_subscription(
            Image, '/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/depth_to_rgb/image_raw', self.depth_callback, 10)
        
        # 发布校正后的图像话题
        self.rgb_pub = self.create_publisher(Image, '/undistorted/rgb/image_raw', 10)
        self.depth_pub = self.create_publisher(Image, '/undistorted/depth_to_rgb/image_raw', 10)
        
        # 相机畸变参数
        self.dist_coeffs = np.array([-9.1e-02, -2.44e+00, 2.57e-04, -3.87e-04,
                                     1.74e+00, -2.1e-01, -2.22e+00, 1.63e+00])
        self.K = np.load('/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/camera_intrinsic1.npy')

    def rgb_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        undistorted_image = self.undistort_image(cv_image)
        undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted_image, encoding='bgr8')
        self.rgb_pub.publish(undistorted_msg)

    def depth_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        undistorted_image = self.undistort_image(cv_image)
        undistorted_msg = self.bridge.cv2_to_imgmsg(undistorted_image, encoding='16UC1')
        self.depth_pub.publish(undistorted_msg)

    def undistort_image(self, image):
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coeffs, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, self.K, self.dist_coeffs, None, newcameramtx)
        return undistorted_image

def main(args=None):
    rclpy.init(args=args)
    undistort_node = UndistortNode()
    rclpy.spin(undistort_node)
    undistort_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
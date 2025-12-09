import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.ndimage import label
from PIL import Image as PILImage
import time


class RealSenseCamera(Node):
    def __init__(self):
        super().__init__('realsense_camera_node')
        
        # 创建 CvBridge 对象，用于将 ROS 图像消息转换为 OpenCV 图像
        self.bridge = CvBridge()

        # 订阅 RealSense 相机的 RGB 和深度图像
        self.rgb_sub = self.create_subscription(Image, "/rgb/image_raw", self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, "/depth_to_rgb/image_raw", self.depth_callback, 10)  # NOTE：must sub the aligned depth image
        self.camera_info_sub = self.create_subscription(CameraInfo, "/rgb/camera_info", self.camera_info_callback, 10)
        self.rgb_image = None
        self.depth_image = None
        self.received_camera_info = False
        self.received_rgb_image = False
        self.received_depth_image = False

        self.K = np.array([
            [608, 0, 642],
            [0, 608, 363.6],
            [0, 0, 1]
        ])
        
        # 畸变系数 D
        self.D = np.array([0, 0, 0, 0, 0])  

        try:
            transform_matrix = np.load("camera_extrinsic1.npy", allow_pickle=True)
            # 假设加载的是一个完整的4x4变换矩阵
            if transform_matrix.shape == (4, 4):
                # 从变换矩阵中提取旋转部分和平移部分
                self.R = transform_matrix[:3, :3]
                self.t = transform_matrix[:3, 3:4]*1000.0  # 保持列向量形式  转换为mm
                self.transform_matrix = transform_matrix  # 保存完整矩阵以供需要时使用
                self.loaded_extrinsics = True
                self.get_logger().info(f"Loaded extrinsics:\nRotation:\n{self.R}\nTranslation:\n{self.t.flatten()}")
            else:
                raise ValueError("Expected a 4x4 transformation matrix")
        except Exception as e:
            self.get_logger().warn(f"Failed to load extrinsics: {e}")
            self.R, self.t = np.eye(3), np.array([[0], [0], [0]])
            self.transform_matrix = np.eye(4)
            self.loaded_extrinsics = False
            
        self.get_logger().info("RealSenseCamera initialized")
        self.get_logger().info(f"Using extrinsics:\nRotation:\n{self.R}\nTranslation:\n{self.t.flatten()}")

    def rgb_callback(self, msg):
        """处理接收到的 RGB 图像消息"""
        if not self.received_rgb_image:
            self.get_logger().info("Received RGB image")
            self.received_rgb_image = True
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def depth_callback(self, msg):
        """处理接收到的深度图像消息"""
        if not self.received_depth_image:
            self.get_logger().info("Received depth image")
            self.received_depth_image = True
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")  # 深度图像是16位无符号整数

    def camera_info_callback(self, msg):
        """处理相机内参"""
        if self.received_camera_info:
            return  # 只处理一次
        self.get_logger().info("Received camera info")
        self.received_camera_info = True
        self.K = np.array(msg.k).reshape(3, 3)
        self.D = np.array(msg.d)  # 畸变系数
        self.get_logger().info(f"Camera intrinsic matrix K:\n{self.K}")
        self.get_logger().info(f"Camera distortion coefficients D:\n{self.D}")

    def capture_image(self, image_type):
        if image_type == "rgb":
            if self.rgb_image is None:
                raise ValueError("RGB image is not available yet!")
            return self.rgb_image
        elif image_type == "depth":
            if self.depth_image is None:
                raise ValueError("Depth image is not available yet!")
            return self.depth_image
        else:
            raise Exception("Invalid image type!")

    def hsv_limits(self, color):
        c = np.uint8([[color]])  # BGR values
        hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

        hue = hsvC[0][0][0]  # Get the hue value

        # Handle red hue wrap-around
        if hue >= 165:  # Upper limit for divided red hue
            lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
            upperLimit = np.array([180, 255, 255], dtype=np.uint8)
        elif hue <= 15:  # Lower limit for divided red hue
            lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
            upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
        else:
            lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
            upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

        return lowerLimit, upperLimit

    def detect_end_effector(self):
        def keep_largest_blob(image):
            # Ensure the image contains only 0 and 255
            binary_image = (image == 255).astype(int)

            # Label connected components
            labeled_image, num_features = label(binary_image)

            # If no features, return the original image
            if num_features == 0:
                return np.zeros_like(image, dtype=np.uint8)

            # Find the largest component by its label
            largest_blob_label = max(range(1, num_features + 1), key=lambda lbl: np.sum(labeled_image == lbl))

            # Create an output image with only the largest blob
            output_image = (labeled_image == largest_blob_label).astype(np.uint8) * 255

            return output_image

        color = [158, 105, 16]

        # Get bounding box around object
        frame = self.capture_image("rgb")
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lowerLimit, upperLimit = self.hsv_limits(color=color)
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        mask = keep_largest_blob(mask)
        mask_ = PILImage.fromarray(mask)
        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imwrite("calib.png", frame)

        return [int((x1 + x2) / 2), int((y1 + y2) / 2)], frame

    def capture_points(self):
        return self.capture_image("depth")

    def pixel_to_3d_points(self):
        depth_pc = self.capture_points()
        K_inv = np.linalg.inv(self.K)

        # Get array of valid pixel locations
        shape = depth_pc.shape
        xv, yv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        nan_mask = ~np.isnan(depth_pc)
        xv, yv = xv[nan_mask], yv[nan_mask]
        pc_all = np.vstack((xv, yv, np.ones(xv.shape)))

        # Convert pixel to world coordinates
        s = depth_pc[yv, xv]
        pc_camera = s * (K_inv @ pc_all)

        # display pc_camera using open3d
        import open3d as o3d
        # pc = o3d.geometry.PointCloud()
        # pc.points = o3d.utility.Vector3dVector(pc_camera.T)
        # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
        # o3d.visualization.draw_geometries([pc, axis], window_name="PointCloud in Camera Frame")

        # pw_final = (R_inv @ (pc_camera - self.t)).T
        pw_final = ((self.R @ pc_camera) + self.t).T  # use R,t to convert to world coordinates
        pw_final = pw_final.reshape(shape[0], shape[1], 3)

        self.get_logger().info(f"Converted pixel to 3D points, shape: {pw_final.shape}")

        # display using o3d
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pw_final.reshape(-1, 3))
        # Create a larger, thicker coordinate frame at the origin (world axis)
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
        # Optionally, set a background color and point size for better visibility
        '''o3d.visualization.draw_geometries(
            [pc, axis],
            window_name="PointCloud with World Axis",
            point_show_normal=False,
            width=1280,
            height=720,
            left=50,
            top=50,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )'''

        return pw_final
    
    def world_to_pixel_coordinates(self, world_coordinates):
        """
        Project a 3D point in base/world frame into pixel coordinates of the color optical frame.

        Args:
            world_coordinates : array-like, shape (3,), meters in base/world frame.

        Uses:
            self.R : (3,3) rotation of base <- optical  (optical -> base)
            self.t : (3,)   translation of base <- optical (meters)
            self.K : (3,3) camera intrinsics for the color optical frame

        Returns:
            (u, v, Z_opt) where u,v are pixel coordinates (floats), Z_opt is depth in meters
            Returns (None, None, Z_opt) if point is behind the camera (Z_opt <= 0) or invalid.
        """
        # Ensure shapes
        R_bo = np.asarray(self.R, dtype=np.float64)           # base <- optical
        t_bo = np.asarray(self.t, dtype=np.float64).reshape(3, 1)
        K     = np.asarray(self.K, dtype=np.float64)

        p_b = np.asarray(world_coordinates, dtype=np.float64).reshape(3, 1)

        # Invert extrinsics analytically to get optical <- base
        # R_ob = R_bo^T ; t_ob = -R_bo^T @ t_bo
        R_ob = R_bo.T
        t_ob = -R_ob @ t_bo

        # Transform base -> optical
        p_opt = R_ob @ p_b + t_ob   # (3,1)
        X, Y, Z = p_opt.flatten()

        # Point behind the camera or invalid
        if not np.isfinite(Z) or Z <= 0:
            self.get_logger().warn(f"Point {world_coordinates} is behind the camera or invalid (Z={Z})")
            return None

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        u = (fx * X / Z) + cx
        v = (fy * Y / Z) + cy

        # Optional: sanity check for NaNs/Infs
        if not (np.isfinite(u) and np.isfinite(v)):
            return None, None, Z
        
        self.get_logger().info(f"Projected world point {world_coordinates} to pixel ({u}, {v}) with depth {Z}m")

        return [round(u), round(v)]
    
    def get_average_depth(self, x, y):
        """根据周围9个点计算深度值的平均值，并去掉无效点。"""
        depth_image = self.capture_points()
        
        # 定义 3x3 邻域
        neighborhood = [
            (dx, dy) for dx in range(-1, 2) for dy in range(-1, 2)
        ]
        
        valid_depths = []
        print(f"Depth image shape: {depth_image.shape}")
        
        # save depth image for debug
        cv2.imwrite("realsense_log/debug_depth.png", depth_image.astype(np.uint16))
        
        # 遍历 3x3 邻域并收集有效的深度值
        for dx, dy in neighborhood:
            nx, ny = x + dx, y + dy
            # print(nx, ny)
            # 确保坐标在图像范围内
            if 0 <= nx < depth_image.shape[1] and 0 <= ny < depth_image.shape[0]:
                # print("in range")
                depth_value = depth_image[ny, nx]
                # print(depth_value)
                # 检查深度值是否有效
                if depth_value > 0 and not np.isnan(depth_value):
                    valid_depths.append(depth_value)
            # print("one point done")
        print(f"Gotten depth values")
        print(valid_depths)
        
        # 如果存在有效的深度值，则计算其平均值
        if valid_depths:
            print(f"Valid depth values in the neighborhood of ({x}, {y})")
            return np.mean(valid_depths)  # 返回平均深度值，形状是 ()
        else:
            # raise ValueError(f"Invalid depth values in the neighborhood of ({x}, {y})")
            print(f"Invalid depth value at ({x}, {y})")
        print("finished")

    def get_camera_coordinates(self, x, y):
        """根据像素坐标转换为相机坐标系中的 3D 坐标。"""
        # 获取深度图像
        depth_value = self.get_average_depth(x, y)  # 使用平均深度值
        
        # 获取深度值
        if depth_value==None or depth_value <= 0 or np.isnan(depth_value):
            # raise ValueError(f"Invalid depth value at ({x}, {y}): {depth_value}")
            self.get_logger().warn(f"Invalid depth value at ({x}, {y}): {depth_value}")
            camera_coordinates= np.array([0, 0, 0])
        else:
            # 计算相机坐标系中的 3D 坐标
            camera_coordinates = (np.linalg.inv(self.K) @ np.array([x, y, 1]) * depth_value).reshape(3, 1)
            self.get_logger().info(f"Camera Coordinates: {camera_coordinates.flatten()}")

        return camera_coordinates

    def get_world_coordinates(self, x, y):
        """根据像素坐标转换为世界坐标系中的 3D 坐标。"""
        # 获取相机坐标系中的 3D 坐标
        camera_coordinates = self.get_camera_coordinates(x, y)
        # print("Camera Coordinates:", camera_coordinates)
        # 使用外参矩阵进行转换
        # print("Rotation Matrix:", self.R)
        # print("Translation Vector:", self.t)
        if camera_coordinates[0]==0 and camera_coordinates[1]==0 and camera_coordinates[2]==0: # 当深度值无效时全部赋值为0
            # raise ValueError(f"Invalid camera coordinates at ({x}, {y}): {camera_coordinates}")
            self.get_logger().warn(f"Invalid camera coordinates at ({x}, {y}): {camera_coordinates}")
            world_coordinates = np.array([0, 0, 0])
        else:
            self.get_logger().info(f"Converting to world coordinates for pixel ({x}, {y}, camera_coordinates: {camera_coordinates.flatten()})")
            # self.get_logger().info(f"Using Rotation Matrix:\n{self.R}")
            # self.get_logger().info(f"Using Translation Vector:\n{self.t}")
            # self.get_logger().info(f"{camera_coordinates - self.t}, shape: {camera_coordinates.shape}")
            # world_coordinates = np.linalg.inv(self.R) @ (camera_coordinates - self.t)
            world_coordinates = (self.R @ camera_coordinates) + self.t
            # self.get_logger().info(f"alternative world coord: {camera_coordinates @ self.R + self.t}")
            # print("World Coordinates:", world_coordinates)
        print("World Coordinates:", world_coordinates.flatten())
        # draw a circle on the rgb image for debug
        if self.rgb_image is not None:
            debug_image = self.rgb_image.copy()
            cv2.circle(debug_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(debug_image, f"World: {world_coordinates.flatten()}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(debug_image, f"Camera coordinate: {camera_coordinates.flatten()}", (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(f"realsense_log/debug_rgb_x{x}_y{y}.png", debug_image)
        return world_coordinates.flatten()  # 返回扁平化的 3D 坐标，形状为 (3,)

    def close(self):
        """Shutdown the ROS2 node"""
        self.destroy_node()
    
    def transform_point(self, camera_coordinates, transform_matrix):
        """
        Transform a point from camera frame to base frame.

        Args:
            camera_coordinates (np.ndarray): shape (3,1) or (3,)
            transform_matrix (np.ndarray): 4x4 homogeneous matrix

        Returns:
            np.ndarray: shape (3,), point in base frame
        """
        # Ensure correct shape
        p_c = np.asarray(camera_coordinates).reshape(3,)
        
        # Convert to homogeneous (4,)
        p_c_h = np.append(p_c, 1.0)
        
        # Transform
        p_b_h = transform_matrix @ p_c_h
        
        # Return as 3D
        return p_b_h[:3] / p_b_h[3]

def main(args=None):
    """Main function to run the camera node"""
    rclpy.init(args=args)
    
    try:
        camera = RealSenseCamera()
        
        while rclpy.ok():
            rclpy.spin_once(camera, timeout_sec=0.5)
            
            # 获取并显示 RGB 和深度图像
            try:
                rgb_image = camera.capture_image("rgb")
                depth_image = camera.capture_image("depth")
                depth = camera.get_world_coordinates(605, 322)
                print(depth)

                # camera_coord = camera.world_to_pixel_coordinates(
                #     [454.43570167, -32.75466571, -17.55472957]
                #     )
                # print(camera_coord)
                # time.sleep(1)

                if rgb_image is not None:
                    cv2.imshow("RGB Image", rgb_image)
                if depth_image is not None:
                    cv2.imshow("Depth Image", depth_image)

                # 按下 'q' 键退出循环
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Exception as e:
                print(e)
                time.sleep(1)
                pass  # Images not available yet
                
    except KeyboardInterrupt:
        pass
    finally:
        # 释放资源
        if 'camera' in locals():
            camera.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

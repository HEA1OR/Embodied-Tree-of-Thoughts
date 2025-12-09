import pyrealsense2 as rs
import numpy as np
import cv2
from scipy.ndimage import label
from PIL import Image


class RealSenseCamera:
    def __init__(self):
        # 配置相机管道
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)  # 启用彩色流
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)   # 启用深度流

        # 启动相机
        self.profile = self.pipeline.start(config)

        # 获取相机内参
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = color_profile.get_intrinsics()
        self.K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]

        ])
        print("相机内参矩阵K:\n", self.K)

        self.D = np.array(self.intrinsics.coeffs)  # 畸变系数
        print("相机畸变系数D:\n", self.D)
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
            print(f"Failed to load extrinsics: {e}")
            self.R, self.t = np.eye(3), np.array([[0], [0], [0]])
            self.transform_matrix = np.eye(4)
            self.loaded_extrinsics = False

    def capture_image(self, image_type):
        # 等待帧
        frames = self.pipeline.wait_for_frames()
        if image_type == "rgb":
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise Exception("Could not get RGB image!")
            color_image = np.asanyarray(color_frame.get_data())
            return color_image
        elif image_type == "depth":
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                raise Exception("Could not get depth image!")
            depth_image = np.asanyarray(depth_frame.get_data())
            return depth_image
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
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imwrite("calib.png", frame)

        return [int((x1 + x2) / 2), int((y1 + y2) / 2)], frame

    def capture_points(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise Exception("Could not get depth image!")
        depth = np.asanyarray(depth_frame.get_data())
        return depth

    def pixel_to_3d_points(self):
        depth_pc = self.capture_points()
        K_inv = np.linalg.inv(self.K)
        R_inv = np.linalg.inv(self.R)

        # Get array of valid pixel locations
        shape = depth_pc.shape
        xv, yv = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        nan_mask = ~np.isnan(depth_pc)
        xv, yv = xv[nan_mask], yv[nan_mask]
        pc_all = np.vstack((xv, yv, np.ones(xv.shape)))

        # Convert pixel to world coordinates
        s = depth_pc[yv, xv]
        pc_camera = s * (K_inv @ pc_all)
        pw_final = (R_inv @ (pc_camera - self.t)).T
        pw_final = pw_final.reshape(shape[0], shape[1], 3)

        return pw_final
    
    def get_camera_coordinates(self,x, y):
        """根据像素坐标转换为相机坐标系中的 3D 坐标。"""
        # 获取深度图像
        depth_image = self.capture_points()

        # 获取深度值
        depth_value = depth_image[y, x]

        # 计算相机坐标系中的 3D 坐标
        camera_coordinates = np.linalg.inv(self.K) @ np.array([x, y, 1]) * depth_value

        return camera_coordinates
    def get_world_coordinates(self, x, y):
        """根据像素坐标转换为世界坐标系中的 3D 坐标。"""
        # 获取相机坐标系中的 3D 坐标
        camera_coordinates = self.get_camera_coordinates(x, y)

        # 使用外参矩阵进行转换
        world_coordinates = self.R @ camera_coordinates + self.t

        return world_coordinates.flatten()

    def close(self):
        self.pipeline.stop()


if __name__ == "__main__":
    # 初始化相机
    camera = RealSenseCamera()

    try:
        while True:
            # 捕获RGB图像
            rgb_image = camera.capture_image("rgb")
            # 显示视频流
             # 捕获深度图像
            depth_image = camera.capture_image("depth")

            cv2.imshow("RGB Image", rgb_image)
           
            # 显示视频流
            cv2.imshow("Depth Image", depth_image)

            # 按下 'q' 键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # 释放资源
        camera.close()
        cv2.destroyAllWindows()


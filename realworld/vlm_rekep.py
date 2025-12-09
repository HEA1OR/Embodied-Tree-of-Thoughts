from rekep.utils import (
    bcolors,
    get_config,
)
from rekep.constraint_generation import ConstraintGenerator
from rekep.keypoint_proposal import KeypointProposer
import time
from realsense_camera_ros import RealSenseCamera
import os
import cv2
import rclpy
from rclpy.node import Node

rclpy.init(args=None)
##############################################################################################
instruction = 'Open the door of the microwave oven'
#instruction = 'reorient the pen and drop it into a holder'
#instruction = 'pick up the holder'
#instruction = 'close the drawer'
#instruction = 'Pick up a tennis ball'
#instruction = 'Put the apple and the pen holder on the drawer, and the apple should be placed in the pen holder'
#instruction = 'Put the apple and the tennis ball in either the drawer or the pen holder, together or separately'
#################################################################################################
camera = RealSenseCamera()
mask_dir = 'mask'  # 假设掩码文件夹是'mask'
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]  # 只读取png文件
mask = []


from rclpy.executors import MultiThreadedExecutor
import threading

camera_node = camera


def main():
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        single_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
        print(f"Loaded mask {mask_file} with shape: {single_mask.shape}")
        # 确保掩码是二值化的
        _, single_mask_bin = cv2.threshold(single_mask, 127, 255, cv2.THRESH_BINARY)

        # 将二值化掩码添加到masks列表
        mask.append(single_mask_bin)
    global_config = get_config(config_path="./rekep/configs/config.yaml")
    constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
    keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
    while camera.rgb_image is None or camera.depth_image is None:
                time.sleep(0.5)
    rgb = camera.capture_image("rgb")
    points = camera.pixel_to_3d_points()
    keypoints,pixels, projected_img = keypoint_proposer.get_keypoints(rgb, points, mask)
    # convert pixels to world coordinate and log
    for idx, pixel in enumerate(pixels):
        world_coord = camera.get_world_coordinates(pixel[1], pixel[0])
        keypoints[idx] = world_coord
        
    print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
    metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
    # metadata = {'init_keypoint_pixels': pixels, 'num_keypoints': len(pixels)}
    rekep_program_dir = constraint_generator.generate(projected_img, instruction, metadata)

# Create a MultiThreadedExecutor and add both nodes
executor = MultiThreadedExecutor()
executor.add_node(camera_node)

# Run perform_task in a separate thread
def run_main_task():
    main()

main_task_thread = threading.Thread(target=run_main_task)
main_task_thread.start()

# Main loop: spin_once to allow callbacks for both nodes
while main_task_thread.is_alive():
    # print("Spinning executor...")
    executor.spin_once(timeout_sec=0.1)

# After perform_task completes, shut down nodes and executor
camera_node.destroy_node()
executor.shutdown()
rclpy.shutdown()
main_task_thread.join()

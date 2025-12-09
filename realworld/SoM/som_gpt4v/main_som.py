import io
import pdb
import torch
import argparse
from PIL import Image
import cv2
from typing import List
from pathlib import Path
import traceback
import shutil
# realsense 相机
#import pyrealsense2 as rs
# semantic sam
from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from semantic_sam.utils.dist import init_distributed_mode
from semantic_sam.utils.arguments import load_opt_from_config_file
from semantic_sam.utils.constants import COCO_PANOPTIC_CLASSES
from task_adapter.semantic_sam.tasks.inference_semsam_m2m_auto import inference_semsam_m2m_auto1
from task_adapter.semantic_sam.tasks.automatic_mask_generator import prompt_switch
from task_adapter.sam.tasks.inference_sam_m2m_interactive import inference_sam_m2m_interactive

from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('coco_2017_train_panoptic')

from scipy.ndimage import label
import numpy as np
from omegaconf import OmegaConf
from gpt4v import request_gpt4v, clear_history
from gpt4v_azure import  request_gpt4v_multi_image_behavior_azure
from mask_filters import get_mask_filter1

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge



'''
build args
'''
semsam_cfg = "./som_gpt4v/configs/semantic_sam_only_sa-1b_swinL.yaml"
seem_cfg = "./som_gpt4v/configs/seem_focall_unicl_lang_v1.yaml"

semsam_ckpt = "./som_gpt4v/swinl_only_sam_many2many.pth"
sam_ckpt = "./som_gpt4v/sam_vit_h_4b8939.pth"
seem_ckpt = "./som_gpt4v/seem_focall_v1.pt"

opt_semsam = load_opt_from_config_file(semsam_cfg)


'''
build model
'''
model_semsam = BaseModel(opt_semsam, build_model(opt_semsam)).from_pretrained(semsam_ckpt).eval().cuda()

def inference(image, granularity, *args, **kwargs):
    # choose model according to segmentation granularity
    if granularity < 1.5:
        model_name = 'seem'
    elif granularity > 2.5:
        model_name = 'sam'
    else:
        model_name = 'semantic-sam'
        if granularity < 1.5 + 0.14:                # 1.64
            level = [1]
        elif granularity < 1.5 + 0.28:            # 1.78
            level = [2]
        elif granularity < 1.5 + 0.42:          # 1.92
            level = [3]
        elif granularity < 1.5 + 0.56:          # 2.06
            level = [4]
        elif granularity < 1.5 + 0.70:          # 2.20
            level = [5]
        elif granularity < 1.5 + 0.84:          # 2.34
            level = [6]
        else:
            level = [6, 1, 2, 3, 4, 5]


    text_size, hole_scale, island_scale=500,100,100
    text, text_part, text_thresh = '','','0.0'
    anno_mode = ['Mask', 'Mark']
    # anno_mode = ['Mark']
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        semantic=False

        if model_name == 'semantic-sam':
            model = model_semsam
            output, mask = inference_semsam_m2m_auto1(model, image, level, text, text_part, text_thresh, text_size, hole_scale, island_scale, semantic, alpha=0.1, anno_mode=anno_mode, *args)
    return output, mask


def gpt4v_response(message, vision, stage):
    for i in range(4):
        try:
            if stage == 1:
                # locate the object in the image
                prompt1 = f"""
Answer the question as if you are a robot with a parallel jaw gripper having access to a segmented photo. Your task is to find the target objects for the given task. Follow the exact format.
The first line should be the objects needed to do the task, which is the target objects, which can be 1 or 2.
The second line should be the labels of the target objects.


Example:
Instruction: pour tea into the cup
Object: teapot, cup
Object Label: [2] [3]

Instruction: {message}
"""
                res = request_gpt4v(prompt1, vision)
                return res
            elif stage == 2:
                # locate the object in the image
                prompt2 = f"""
Now look at the center cropped image of the target object. Which labeled component should you grasp? 
First describe the object part each label corresponds to. Then output the label of the target object part.
ATTENTION: you act as the robot, not the person in the scene. You should avoid fragile parts, and hold on protective covers if there exists. The output label must be part of the target object.
EXAMPLE:
Label 1: the handle of the mug
Label 2: the body of the mug
Label 3: the water in the mug
Target Object Part: [1]

Instruction: {message}
"""
        except Exception as e:
            traceback.print_exc()
            continue    # try again

    
def extract_label(respond) -> List[int]:
    '''Extract the label in the respond of GPT-4V'''
    # 将响应按空格分割
    respond = respond.split(' ')
    # 去除多余字符
    respond = [r.replace('.', '').replace(',', '').replace(')', '').replace('"', '') for r in respond]
    
    # 提取方括号内的内容
    labels = []
    for r in respond:
        if '[' in r and ']' in r:
            # 提取方括号中的数字并转换为整数
            content = r.split('[')[1].split(']')[0]
            labels.extend(content.split(','))
    
    # 保留纯数字并去重（保持顺序）
    labels = [int(r) for r in labels if r.isdigit()]
    labels = list(dict.fromkeys(labels))  # 使用有序去重
    
    return labels


def get_mask(masks, labels):
    # 获取所有对应的 mask
    selected_masks = [masks[int(label)-1] for label in labels]  # 遍历 labels
    return selected_masks


def crop_mask(image, output, mask):
    output_img = Image.fromarray(output)
     # center crop the mask
    mask_bbox1 = (np.array(mask['bbox']) * (image.width / output_img.width)).astype(int)
    print(mask_bbox1)
    margin = 0.1
    wh = (image.height, image.width)
    bbox = (int(np.clip(mask_bbox1[0]-margin*mask_bbox1[2], 0, wh[1])), 
            int(np.clip(mask_bbox1[1]-margin*mask_bbox1[3], 0, wh[0])), 
            int(np.clip(mask_bbox1[0]+(1+margin)*mask_bbox1[2], 0, wh[1])), 
            int(np.clip(mask_bbox1[1]+(1+margin)*mask_bbox1[3], 0, wh[0])))
    image1 = image.crop(bbox)
    mask_in_bbox = Image.fromarray(mask['segmentation']).resize(image.size).crop(bbox)
    mask_in_bbox.save('outputs/mask_in_bbox.png')
    return image1, bbox, np.array(mask_in_bbox)

def combine_mask(image, mask, bbox):
    mask_to_save = np.zeros((image.height, image.width), dtype=np.uint8)
    img_mask = Image.fromarray(mask['segmentation'])
    resized_mask = img_mask.resize((bbox[2]-bbox[0], bbox[3]-bbox[1]))
    mask_to_save[bbox[1]:bbox[3], bbox[0]:bbox[2]] = np.array(resized_mask)
    return mask_to_save

# realsense camera python sdk
# def capture_rgb_image(output_path):   
#     # 配置Realsense管道
#     pipeline = rs.pipeline()
#     config = rs.config()
#     config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

#     try:
#         # 启动管道
#         pipeline.start(config)

#         # 等待相机稳定
#         for _ in range(30):
#             pipeline.wait_for_frames()

#         # 获取一帧RGB图像
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()

#         if not color_frame:
#             raise RuntimeError("未能捕获到RGB图像")

#         # 将图像转换为numpy数组
#         color_image = np.asanyarray(color_frame.get_data())

#         # 保存图像到指定路径
#         cv2.imwrite(output_path, color_image)
#         print(f"RGB图像已保存到: {output_path}")

#     finally:
#         # 停止管道
#         pipeline.stop()

# realsense camera ros2
# RealSense 相机类
class RealSenseCamera(Node):
    def __init__(self):
        super().__init__('realsense_camera_node')
        # 初始化 CvBridge
        self.bridge = CvBridge()        
        # 订阅图像话题
        self.color_sub = self.create_subscription(ROSImage, '/rgb/image_raw', self.color_callback, 10)       
        self.color_image = None
        # 创建显示窗口
        cv2.namedWindow("Color Image", cv2.WINDOW_AUTOSIZE)
        self.get_logger().info("RealSense camera initialized.")

    def color_callback(self, msg):
        try:
            # 将 ROS 图像消息转换为 Numpy 数组
            color_image_bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 存储 RGB 格式用于处理
            self.color_image = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Debug: 检查图像数据
            self.get_logger().info(f"Received image: {color_image_bgr.shape}, min: {color_image_bgr.min()}, max: {color_image_bgr.max()}")
            
            # 显示 BGR 格式的图像
            cv2.imshow("Color Image", color_image_bgr)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Failed to convert color image: {e}")

    def depth_callback(self, msg):
        try:
            # 将ROS深度图像消息转换为Numpy数组
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file name')
    cli_args = parser.parse_args()
    
    output_path = "grasp/color2.png"
    # capture_rgb_image(output_path)
    
    # Initialize ROS2
    rclpy.init()
    camera = RealSenseCamera()
    
    # save camera.color_image to output_path
    while camera.color_image is None:
        camera.get_logger().info("Waiting for camera image...")
        rclpy.spin_once(camera, timeout_sec=0.1)
    # 保存图像 - 将 RGB 转换回 BGR 用于保存
    color_image_bgr_to_save = cv2.cvtColor(camera.color_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, color_image_bgr_to_save)

    try:
        while True:
            input("Press enter to inference ;")
            args = OmegaConf.load(f'{cli_args.config}/grasp_cfg.yaml')
            image = Image.open(args.image)
            shutil.copyfile(args.image, f'{cli_args.config}/image.png')
            output, masks = inference(image, args.granularity1)
            Image.fromarray(output).save(f'{cli_args.config}/som.png')
            _ = input("Press enter to ask GPT-4V ;")
            respond = gpt4v_response(args.task, Image.fromarray(output), 1)
            print(respond)
            if respond is None:
                clear_history()
                continue
            labels = extract_label(respond)
            print(labels)
            mask = get_mask(masks, labels)
            # 保存所有mask图片
            for i, m in enumerate(mask):
                resized_mask = Image.fromarray(m['segmentation']).resize((1280, 720))
                resized_mask.save(f'/home/dell/workspace/xwj/Enhanced_ReKep4xarm_Tinker-ros2_migration/mask/mask_{i}.png')
                print(f'mask_{i}.png saved')
            clear_history()
            # Ask user if they want to continue
            cont = input("Do you want to continue? (y/n): ")
            if cont.lower() != 'y':
                break
    finally:
        camera.destroy_node()
        rclpy.shutdown()

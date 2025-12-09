import torch
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

# 加载 DINOv2 模型
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model.eval()

# 图像预处理函数
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    return preprocess(frame_pil)

def process_frame_with_dino(frame, model):
    # 预处理帧
    input_tensor = preprocess_frame(frame).unsqueeze(0)  # 添加 batch 维度
    with torch.no_grad():
        feature = model(input_tensor)  # 提取特征
    return feature.cpu().numpy()

def main():
    # 从摄像头捕获视频（可以将参数换为视频文件路径）
    cap = cv2.VideoCapture(0)  # 0 表示使用默认摄像头，换为视频文件路径可处理视频文件
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 获取视频的宽度、高度
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # 定义视频编码器并创建 VideoWriter 对象保存结果
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))

    while True:
        # 逐帧捕获
        ret, frame = cap.read()

        if not ret:
            print("无法接收帧（流结束？）")
            break

        # 使用 DINOv2 提取特征
        feature = process_frame_with_dino(frame, model)

        # 在帧上显示一些处理后的信息，例如在左上角显示 "Processed" 字样
        cv2.putText(frame, 'Processed', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 实时显示结果帧
        cv2.imshow('Processed Video', frame)

        # 将帧写入视频文件
        out.write(frame)

        # 按 'q' 键退出循环
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # 释放视频捕获和写入对象
    cap.release()
    out.release()

    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

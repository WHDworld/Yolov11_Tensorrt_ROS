from ultralytics import YOLO
import time
# Load a YOLO11n PyTorch model
model = YOLO("./yolov11n_v3.pt")

# Export the model to TensorRT
model.export(
    imgsz=(640, 640),        # 输入分辨率
    dynamic=False,            # 启用动态维度 (batch, height, width)
    simplify=True,           # 简化模型结构
    format="onnx", 
    device="0", 
    half=False)
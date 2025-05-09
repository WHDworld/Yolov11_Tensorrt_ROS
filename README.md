Yolov11_Tensorrt_ROS: YOLOv11 与 TensorRT 在 ROS 中的实现
Yolov11_Tensorrt_ROS 是一个基于 ROS 框架的实时目标检测系统，结合了 YOLOv11 的高精度检测能力和 TensorRT 的高性能推理优化。该系统支持多种精度模式（FP32/FP16/INT8），能够在 NVIDIA GPU 上实现高效的目标检测任务。
主要特点
高性能推理：利用 TensorRT 优化 YOLOv11 模型，显著提升检测速度
多精度支持：支持 FP32、FP16 和 INT8 量化模式，平衡精度与速度
ROS 集成：完全集成 ROS 框架，支持图像订阅和检测结果发布
灵活配置：通过配置文件轻松调整模型参数和检测阈值
自定义 Backbone：采用 ShuffleNetV2 作为骨干网络，轻量且高效
系统架构
系统主要包含以下几个部分：
ROS 节点：负责图像数据的接收和处理结果的发布
TensorRT 引擎：负责模型的优化和推理
后处理模块：处理检测结果，包括 NMS 和边界框解析
可视化模块：将检测结果可视化并发布
安装指南
依赖环境
Ubuntu 18.04/20.04
ROS Melodic/Noetic
CUDA 10.2/11.x
cuDNN 8.x
TensorRT 8.x
OpenCV 4.x
PyTorch 1.8+
编译步骤
创建并初始化 catkin 工作空间：
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/your_username/Yolov11_Tensorrt_ROS.git
cd ..
catkin_make
source devel/setup.bash
下载预训练模型：
将 YOLOv11 预训练模型 (.onnx 格式) 放入指定目录，并在配置文件中指定路径。
配置参数：
编辑配置文件config/config.yaml，设置模型路径、检测阈值和其他参数。
使用方法
运行检测节点
cd Yolov11_Tensorrt_ROS
source ./start.sh
参数配置
主要配置参数位于config/config.yaml文件中：
yolov11_tensorrt_config:
  trtFile: "path/to/your/model.trt"  # TensorRT引擎文件路径
  onnxFile: "path/to/your/model.onnx"  # ONNX模型文件路径
  kGpuId: 0  # GPU设备ID
  kNmsThresh: 0.45  # NMS阈值
  kConfThresh: 0.25  # 置信度阈值
  kNumClass: 80  # 类别数量
  kInputH: 640  # 输入图像高度
  kInputW: 640  # 输入图像宽度
  use_FP16_Mode: 1  # 是否使用FP16模式
  use_INT8_Mode: 0  # 是否使用INT8模式
  cacheFile: "path/to/calibration.cache"  # INT8校准缓存文件
  calibrationDataPath: "path/to/calibration/images"  # 校准图像路径
  vClassNames: ["person", "bicycle", "car", ...]  # 类别名称列表
发布的话题
/yolo_detector/detected_image: 可视化后的检测结果图像
/yolo_detector/detected_bounding_boxes: 检测到的边界框信息
/yolo_detector/yolo_time: 检测耗时信息
性能优化
系统支持三种精度模式：
FP32：完整精度模式，提供最高检测精度
FP16：半精度模式，在保持较高精度的同时显著提升推理速度
INT8：8 位整型量化模式，极致性能优化，适合边缘设备
通过配置文件中的use_FP16_Mode和use_INT8_Mode参数可以轻松切换精度模式。

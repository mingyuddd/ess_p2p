# 自动驾驶轮椅感知与控制系统
基于 ROS 2 + ZED 双目相机 + TensorRT 的自动驾驶轮椅感知控制平台，具备实时语义分割巡线、点云障碍物避障、LQR 闭环控制等能力。

## 系统架构

ZED 双目相机
    │
    ├─ RGB 图像（GPU） ──→ TensorRT BiSeNet 推理 ──→ 语义 Mask
    │                                                    │
    └─ 点云 XYZRGBA（GPU） ──→ CUDA 滤波与分类 ─────────────┤
                                                         │
                                              FusionNode（感知）
                                                         │
                                            perception/output 话题
                                                         │
                                           ControllerNode（控制）
                                                         │
                                                   cmd_vel 话题
                                                         │
                                               底层蓝牙驱动 → 电机

## 工程
两个节点以 **ROS 2 Composable Node** 形式运行在同一容器（`wheel_container`）内，通过 **Intra-Process Communication** 实现零拷贝消息传递，降低通信延迟。`FusionNode` 采用 **ROS 2 Lifecycle** 管理，确保相机和推理引擎完全初始化后才进入激活状态。

wheel_cuda/
├── src/
│   ├── wheel_perception/          # 核心感知与控制包
│   │   ├── config/
│   │   │   ├── params.yaml        # 感知算法参数（ZED、AI、ROI、障碍物）
│   │   │   └── wheel_config.yaml  # LQR 控制参数与逻辑状态机参数
│   │   ├── include/wheel_perception/core/
│   │   │   ├── zed_driver.hpp     # ZED SDK 封装（Pimpl）
│   │   │   ├── ai_engine.hpp      # TensorRT 推理引擎封装
│   │   │   ├── obstacle_fusion.hpp# GPU 点云滤波 + 语义融合
│   │   │   └── lqr_controller.hpp # LQR 横向控制器
│   │   ├── src/
│   │   │   ├── fusion_node.cpp    # 感知主节点（Lifecycle）
│   │   │   ├── controller_node.cpp# 控制主节点
│   │   │   └── core/
│   │   │       ├── zed_driver.cpp
│   │   │       ├── ai_engine.cpp
│   │   │       ├── obstacle_fusion.cu  # CUDA 点云滤波 Kernel
│   │   │       └── preprocess.cu       # CUDA 图像预处理 Kernel
│   │   │       └── global_flags.cpp    # 需要添加全局参数或者标志位在此添加
│   │   └── launch/
│   │       └── run_launch.py      # 一键启动（含生命周期管理）
│   │
│   ├── wheel_msgs/                # 自定义消息包
│   │   └── msg/
│   │       ├── PerceptionOutput.msg  # 感知输出（距离、偏角、障碍物列表）
│   │       └── ObstacleInfo.msg      # 单个障碍物信息
│   │
│   └── vision_opencv/             # cv_bridge（cv::Mat ↔ ROS Image 转换）
│
├── BiSeNet/                       # 语义分割模型训练代码
├── opencv/                        # OpenCV 4.13.0 源码与编译产物


---

## 功能模块说明

### 1. ZED 驱动（`ZedDriver`）
同时通过 ZED **Positional Tracking**（融合 IMU）提供六自由度位姿，以四元数形式输出
坐标系约定（全系统统一）：

X 轴 → 朝前（Forward）
Y 轴 → 朝左（Left）
Z 轴 → 朝上（Up
HD720 分辨率下相机内参通过 `getCameraParams()` 在运行时动态获取（`fx, fy, cx, cy, baseline`）。

### 2. TensorRT 推理引擎（`AiEngine`）
加载预先转换好的 `.engine` 文件，执行以下流程：
1. **CUDA 预处理 Kernel（`preprocess.cu`）**：将 ZED GPU 图像缩放并归一化至 BiSeNet 输入尺寸（如 1024×512），全程在 GPU 完成，避免 CPU 数据搬运。
2. **TensorRT 异步推理**：调用 `enqueueV2` 在 CUDA Stream 上执行推理。
3. **输出**：逻辑值张量（NCHW，`float32`），送入 `ObstacleFusion::processSegmentation` 做 ArgMax。

### 3. GPU 点云融合（`ObstacleFusion`，CUDA）
所有运算在 GPU 上完成，包含以下 CUDA Kernel：
| Kernel | 功能 |
|---|---|
| `voxel_filter_kernel` | 自适应体素网格降采样 + 双类别标记（矩形巡线区 / 椭圆危险区） |
| `extract_right_edge_kernel` | 结合语义 Mask 与点云，逐行提取右侧道路边界三维点 |
| `launch_argmax_kernel` | BiSeNet 输出 Float 张量 → 逐像素 ArgMax → `uint8` Mask |
**双类别几何判据：**
- **类别 1（矩形区）**：`min_x ≤ X ≤ max_x` 且 `Y ≥ min_y`，用于提供 LQR 右侧边界点云
- **类别 2（椭圆区）**：`ellipse_x × X² + ellipse_y × Y² < thres`，用于紧急避障触发

### 4. LQR 横向控制器（`LqrController`）
采用线性二次调节器（LQR）对轮椅横向偏差进行闭环控制，状态量为：
[横向偏差 e,  横向偏差变化率 ė,  航向偏差 θ_e,  航向偏差变化率 θ̇_e]
控制输出（转向角增量）经协议增益 `lqr.gain = 60.0` 放大后，叠加到底层基准转向指令上发送给轮椅驱动。

### 5. 控制状态机（`ControllerNode`）
订阅 `perception/output` 话题，实现以下状态切换：

```
CRUISE（正常巡线）
    │ 障碍物进入危险区（min_distance < stop_dist）
    ↓
AVOID_HARD_LEFT / AVOID_HARD_RIGHT（强烈转向避让）
    │ 
    ↓
CONTINUE_STRAIGHT（继续直行）
    │ 里程计确认已通过障碍物（pass_clearance）
    ↓
RECOVER_TURN（回正恢复，持续 recover_time 秒）
    ↓
CRUISE
    │ 无有效道路边界
    ↓
STOP（停车等待）
```

## 话题与消息

| 话题 | 类型 | 方向 | 说明 |
| `perception/output` | `wheel_msgs/PerceptionOutput` | FusionNode → ControllerNode | 距离、偏角、障碍物列表 |
| `cmd_vel` | `geometry_msgs/Twist` | ControllerNode → 底层 | 速度与转向指令 |
| `/odom` | `nav_msgs/Odometry` | FusionNode → ControllerNode | ZED 视觉里程计（含 IMU 融合） |
| `zed/point_cloud` | `sensor_msgs/PointCloud2` | FusionNode → Rviz | 全量滤波点云（调试） |
| `perception/debug/cloud_rect` | `sensor_msgs/PointCloud2` | FusionNode → Rviz | 矩形巡线区点云（调试） |
| `perception/debug/cloud_ellipse` | `sensor_msgs/PointCloud2` | FusionNode → Rviz | 椭圆避障区点云（调试） |
| `debug/viz` | `sensor_msgs/Image` | FusionNode → Rviz | 语义分割可视化图像 |

### 编译
cd /home/smy/alpha_ws
colcon build

# 仅编译感知与消息包(这个一般不用)
colcon build --packages-select wheel_msgs wheel_perception \
  --cmake-args -DWHEEL_CUDA_ARCHITECTURES=72

source install/setup.bash

### 模型转换（`.pth` → `.engine`）

```
1.先修改pth2engine.py文件内的输入与输出路径
cd src/wheel_perception/config
python3 pth2engine.py
```

### 启动

```
source install/setup.bash
# 确保 ZED 相机已连接
ros2 launch wheel_perception run_launch.py
```
启动脚本自动完成以下操作：
1. 发布 `base_link → zed_left_camera_frame` 静态 TF 变换
2. 在 `wheel_container` 内加载 `FusionNode` 与 `ControllerNode`
3. 延迟 3s 后发送 `configure` 指令（初始化相机与推理引擎）
4. 延迟 8s 后发送 `activate` 指令（开始感知循环，约 60Hz）

### 回放数据集模式选择
在原有的基础上添加了数据集的回放功能
# 使用
1. 修改params.yaml内的 use_dataset_mode 这个参数，当它为true时即可进行该模式（注意重新编译）。
2. python3 zed_dataset_player.py --dataset /home/smy/alpha_ws/text1 --fps 15 --loop
（数据集的路径自行修改）
3. ros2 launch wheel_perception run_launch.py

### 代码修改
1. 如需添加其它功能，不建议直接对代码的主函数进行修改，建议包装成函数形式，方便调用与禁止
2. 先在自己电脑上进行修改完毕，不建议直接在主板上进行修改，防止影响其他人的代码
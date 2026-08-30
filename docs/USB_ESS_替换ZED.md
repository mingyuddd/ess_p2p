# USB 双目相机 + NVIDIA ESS 深度引擎 替代 ZED —— 集成改动说明

> 结论先行：你要做的三件事——**① 引入 USB/ESS 的 core 源文件 ② 准备 ESS engine 文件 ③ 改 fusion_node.cpp 接口 + params.yaml 参数——全部已经完成**。
> 本文档记录每一步改了什么、为什么这么改，供你在新机器（如 Jetson Orin）上照着重做，也便于排查问题。

---

## 1. 架构对比

```
【旧路线：ZED SDK】
ZED 相机 ── ZED SDK (sl_zed) ──> 左目RGB(GPU) + 深度(GPU) + 点云(GPU) ──> ObstacleFusion ──> LQR 控制
                                    （ZED SDK 内部自己做双目匹配）

【新路线：USB 双目 + ESS】
USB 双目相机 (3840x1080 YUYV) ── V4L2 ──> UsbStereoDriver
    ├── 左目校正图 (BGRA, GPU) ──────────────┐
    └── 右目校正图 (BGRA, GPU) ──> EssDepthEngine (TensorRT 插件) ──> 视差+深度+点云 (GPU)
                                                                          │
                                                    ObstacleFusion ──> LQR 控制（完全复用）
```

两路模式的**最终产物是同一套东西**：左目 RGB + float 深度 + float4 点云，全部在 GPU 显存里。
所以下游感知/控制代码一行不用改。

---

## 2. 为什么下游不用改：统一的 `ZedGpuFrame`

[zed_driver.hpp](../src/wheel_perception/include/wheel_perception/core/zed_driver.hpp) 里定义了统一的帧结构，ZED 和 USB+ESS 都往里面填：

| 字段 | ZED 模式（谁填） | USB+ESS 模式（谁填） |
|---|---|---|
| `rgb_ptr_dev` / `rgb_step_bytes` | ZedDriver::grab | UsbStereoDriver::grab（左目校正图） |
| `right_ptr_dev` / `right_step_bytes` | 不使用 | UsbStereoDriver::grab（右目校正图） |
| `depth_ptr_dev` / `depth_step_bytes` | ZED SDK 深度 | EssDepthEngine::infer 输出 |
| `cloud_ptr_dev` / `cloud_step_bytes` | ZED SDK 点云 | EssDepthEngine::infer 输出 |
| `width` / `height` | ZED 分辨率 | USB 输出分辨率 (1280x720) |

`update_loop()` 里两路的区别只在"取帧"这一段（见 4.2），取完帧之后 `fusion_->filterCloud(cloud_ptr_dev, ...)`、
`extractRoadEdges(...)` 等全部无感复用。**这就是"只改 rgb 与深度图接口"的本质：改的是帧的"生产端"，不是"消费端"。**

---

## 3. 文件清单

### 3.1 新增文件（USB/ESS 路线专属）

| 文件 | 职责 |
|---|---|
| [usb_stereo_driver.cpp](../src/wheel_perception/src/core/usb/usb_stereo_driver.cpp) + [usb_stereo_driver.hpp](../src/wheel_perception/include/wheel_perception/core/usb_stereo_driver.hpp) | V4L2 抓流、左右目裁剪、OpenCV 校正（stereoRectify）、CUDA 上传 GPU，输出 BGRA 左右目 |
| [ess_depth_engine.cpp](../src/wheel_perception/src/core/usb/ess_depth_engine.cpp) + [ess_depth_engine.hpp](../src/wheel_perception/include/wheel_perception/core/ess_depth_engine.hpp) | dlopen 加载 ESS 插件 → 反序列化 engine → preprocess kernel → enqueueV3 → 视差转深度转点云 |
| [ess_preprocess.cu](../src/wheel_perception/src/core/usb/ess_preprocess.cu) | BGRA→RGB 的 CUDA 预处理 kernel（喂给 ESS 的输入布局） |
| [depth_to_cloud.cu](../src/wheel_perception/src/core/usb/depth_to_cloud.cu) | disparity → depth（`depth = fx * baseline / disparity`）→ float4 点云 的 CUDA kernel |
| [ess/](../src/wheel_perception/ess/) | ESS 相关资产：`light_ess.engine`（深度引擎）、`combined4.engine`（AI 分割引擎）、`plugins/x86_64/ess_plugins.so`（TRT 插件）、`build_engine_trt10.cpp` + `build_all_trt10.sh`（engine 转换脚本）、`test_ess_infer.cpp`（最小推理测试） |
| [camera/stereo_calibration.json](../src/wheel_perception/camera/stereo_calibration.json) | USB 双目相机的标定文件（内参 fx/fy/cx/cy、畸变、baseline），由标定程序生成 |

### 3.2 修改文件

| 文件 | 改动 |
|---|---|
| [fusion_node.cpp](../src/wheel_perception/src/fusion_node.cpp) | 见第 4 节 |
| [params.yaml](../src/wheel_perception/config/params.yaml) | 见第 5 节 |
| [CMakeLists.txt](../src/wheel_perception/CMakeLists.txt) | 见第 6 节 |

### 3.3 未变动的文件（ZED 代码保留但运行时不用）

- `zed_driver.cpp`：仍编译进 wheel_core，但 `camera.type: "usb"` 时不会实例化。
- 感知/控制层：`obstacle_fusion.cu`、`ai_engine.cpp`、`controller_node.cpp` 一行没改。

---

## 4. fusion_node.cpp 的改动详解

### 4.1 `init_camera()` —— 按 `camera.type` 创建驱动

```cpp
if (camera_type_ == "usb") {
    // 硬校验：USB 模式必须用 ESS 出深度
    if (depth_source_ != "ess") { ...return false; }

    usb_driver_ = std::make_unique<core::UsbStereoDriver>();
    if (!usb_driver_->open(usb_cfg)) { ...return false; }

    ess_engine_ = std::make_unique<core::EssDepthEngine>(ess_cfg);
    if (!ess_engine_->initialize()) { ...return false; }

    return true;
}
if (camera_type_ != "zed") { ...return false; }
driver_ = std::make_unique<core::ZedDriver>();
return driver_->open(zed_cfg);
```

### 4.2 `update_loop()` —— 取帧分叉（唯一的分叉点）

```cpp
if (!use_dataset_mode_) {
    if (camera_type_ == "usb") {
        if (!usb_driver_ || !usb_driver_->grab(frame)) return;   // V4L2 抓帧 + 校正 + 上 GPU
        if (!run_ess_depth(frame)) return;                        // ESS 深度推理
    } else {
        if (!driver_ || !driver_->grab(frame)) return;            // ZED 直取
        publish_odometry(frame);                                  // odom 是 ZED 独有
    }
}
// 之后的 AI 分割、点云滤波、车道线、避障全部与模式无关
```

### 4.3 `run_ess_depth()` —— 把 ESS 输出填回 frame

```cpp
bool run_ess_depth(core::ZedGpuFrame& frame) {
    // 1. 取 USB 相机内参（标定文件加载的）
    const auto params = usb_driver_->getCameraParams();
    core::EssDepthEngine::StereoCalibration calibration;
    calibration.fx = params.fx; calibration.fy = params.fy;
    calibration.cx = params.cx; calibration.cy = params.cy;
    calibration.baseline_m = params.baseline;   // 关键：深度 = fx*baseline/disparity

    // 2. 推理（入参是左右目 BGRA GPU 指针）
    core::EssDepthEngine::InferenceOutput output;
    if (!ess_engine_->infer(frame.rgb_ptr_dev, frame.rgb_step_bytes,
                            frame.right_ptr_dev, frame.right_step_bytes,
                            frame.width, frame.height, calibration, output)) {
        return false;
    }
    if (!ess_engine_->waitForCompletion()) return false;

    // 3. 把深度图和点云指针挂到 frame 上，下游无感
    frame.depth_ptr_dev = const_cast<float*>(output.depth_ptr_dev);
    frame.cloud_ptr_dev = const_cast<float4*>(output.cloud_ptr_dev);
    return true;
}
```

### 4.4 本次顺手修复：USB 模式下的可视化空指针（2026-08-19）

`publish_visualization()` 原代码只处理了 ZED 驱动，USB 模式下只要有 rviz 订阅 `debug/viz` 就会空指针崩溃：

```cpp
// 修复前（USB 模式下 driver_ == nullptr → 崩溃）
if (!use_dataset_mode_) {
    params = driver_->getCameraParams();
}

// 修复后：优先 ZED，其次 USB，都没有（数据集模式）用默认
if (driver_) {
    params = driver_->getCameraParams();
} else if (usb_driver_) {
    const auto p = usb_driver_->getCameraParams();
    params.fx = p.fx; params.fy = p.fy; params.cx = p.cx; params.cy = p.cy;
    params.baseline = p.baseline;
} else { ... }
```

同时 `retrieveImage()` 也补了 `usb_driver_` 分支，USB 模式下调试图也能显示了。

---

## 5. params.yaml 的改动详解

### 5.1 模式开关（最重要的一行）

```yaml
camera:
  type: "usb"     # zed | usb —— 整条路线的总开关
```

### 5.2 USB 相机段

```yaml
usb:
  device: "/dev/v4l/by-id/usb-Generic_USB_Camera_YHTek-video-index0"  # 用 by-id 符号链接，插拔后不变
  fourcc: "MJPG"   # [实测教训] 用 MJPG 不要用 YUYV！3840x1080 YUYV@30 = 249 MB/s 未压缩裸流，
                   # 廉价 UVC 芯片（YHTek/Realtek 0bda:5883）持续高压下会固件卡死 →
                   # OpenCV select() 5 秒超时 + "usb frame unavailable"。MJPG 硬件压缩后
                   # 只有 ~30-60 MB/s，稳定出帧。质量影响可忽略（ESS 输入还会缩到 480x288）
  capture_width: 3840     # 相机原始分辨率：左(1920x1080)+右(1920x1080) 拼接
  capture_height: 1080
  fps: 30.0
  buffer_size: 1
  output_width: 1280      # 校正后的单目分辨率（标定文件决定，别乱改）
  output_height: 720
  calibration_json: "src/wheel_perception/camera/stereo_calibration.json"  # 标定文件（相对 ~/wheel_ess 的路径）
  swap_left_right: true   # 相机左右目物理顺序（以你的装配为准）
  require_calibration_size: true   # 要求校正尺寸与标定文件一致
  rectify_alpha: 0.0      # 校正裁剪系数：0=全部裁掉黑边，1=保留全部黑边
  auto_reconnect: true    # USB 断开自动重连
  reconnect_interval_ms: 500
  reconnect_warmup_frames: 3
  controls: { ... }       # V4L2 控件（-1 = 不动），调亮度/曝光等
```

### 5.3 ESS 段

```yaml
ess:
  engine_path: "src/wheel_perception/ess/light_ess.engine"          # ESS 深度引擎（TRT 序列化）
  plugin_path: "src/wheel_perception/ess/plugins/x86_64/ess_plugins.so"
  #        ↑ 注意架构：x86_64 是给 PC 的；Jetson（aarch64）必须改成 aarch64/ess_plugins.so
  input_width: 480        # ESS 网络输入分辨率（engine 固定，别乱改）
  input_height: 288
  confidence_threshold: 0.5   # 视差置信度阈值，滤掉低置信度像素
  min_depth_m: 0.3        # 深度范围（米），与 zed.depth_min/max 对应
  max_depth_m: 10.0
```

### 5.4 ZED 段中仍然生效的项

```yaml
zed:
  depth_source: "ess"     # USB 模式硬性要求 = "ess"
  use_dataset_mode: false # false=直驱相机；true=订阅 /rgb_image、/depth_image 话题（python 节点发图模式）
```

其余 zed.*（resolution_id、fps、depth_mode、svo_path 等）在 USB 模式下**不生效**，可留着不动。

### 5.5 感知/控制参数（perception/lqr/logic 段）

与相机路线无关，**完全没改**，沿用 ZED 时代的值（ROI、椭圆阈值、LQR 增益等）。

---

## 6. CMakeLists.txt 的改动详解

```cmake
# ① wheel_core 源文件里注册 USB/ESS 模块
add_library(wheel_core SHARED
  ...
  src/core/usb/usb_stereo_driver.cpp
  src/core/usb/ess_depth_engine.cpp
  src/core/usb/ess_preprocess.cu
  src/core/usb/depth_to_cloud.cu
  ...
)

# ② TensorRT 10 查找（CUDA 12.2 targets 目录）
set(TRT_ROOT "/usr/local/cuda-12.2/targets/x86_64-linux")
find_library(NVINFER_LIB nvinfer PATHS ${TRT_ROOT}/lib NO_DEFAULT_PATH)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS ${TRT_ROOT}/lib NO_DEFAULT_PATH)

# ③ 【关键坑】include 顺序：TRT 10 头文件必须最前
include_directories(
  ${TRT_ROOT}/include    # 必须在 ${CUDA_INCLUDE_DIRS} 前面！
  include
  ${CUDA_INCLUDE_DIRS}   # cuda-11.8 的 targets/include 里还藏着 TRT 8.6 的 NvInfer.h，
  ...                    # 顺序反了 NV_TENSORRT_MAJOR 会编译成 8，运行直接崩
)

# ④ jsoncpp：usb_stereo_driver 解析标定文件要用，不链接会 symbol lookup error
find_library(JSONCPP_LIB NAMES jsoncpp)
target_link_libraries(wheel_core ... ${JSONCPP_LIB} ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB} ...)
```

---

## 7. 构建与运行（全流程）

```bash
# 1. 加载 TRT 10 + CUDA 12 运行库环境（含 LD_LIBRARY_PATH）
source ~/wheel_ess/src/wheel_perception/ess/trt10_env.sh

# 2. 编译
cd ~/wheel_ess
colcon build --packages-select wheel_perception --cmake-args -DCMAKE_BUILD_TYPE=Release

# 3. source 工作空间（trt10_env.sh 与 setup.bash 都要，各管各的：
#    trt10_env.sh → 动态库搜索路径；setup.bash → ROS 包发现）
source install/setup.bash

# 4. 启动（注意：launch 文件里读的是 install/share 下的 params.yaml，
#    改完 config/params.yaml 必须重新 colcon build 才会生效）
ros2 launch wheel_perception run_launch.py
```

若 `ros2 node list` 看不到 /fusion_node，先 `ros2 daemon stop && ros2 daemon start`。

---

## 8. 当前状态与已知限制

| 项目 | 状态 |
|---|---|
| USB 相机采集/校正 | ✅ 正常（3840x1080@30 **MJPG** → 1280x720 rectified；YUYV 实测会固件卡死/超时，见 8.2） |
| ESS 引擎加载 | ✅ 正常（4 tensor：input_left/input_right [1x3x288x480]、output_left/output_conf [1x288x480]） |
| AI 分割引擎加载 | ✅ 正常 |
| "Perception Configured" | ✅ 正常 |
| **ESS 深度推理（execute）** | ❌ **GTX 1660 SUPER 上失败**（详见 8.1） |
| USB 模式可视化 | ✅ 已修复（4.4 节） |

### 8.1 ESS 深度推理在 GTX 1660 SUPER 上失败的原因（硬件限制）

- ESS 插件的内核需要 **>64KB 动态共享内存**，而 GTX 1660 SUPER（Turing sm_75）每 block 共享内存**物理上限 64KB**；
- `cudaFuncSetAttribute` 申报超限 → `cudaErrorInvalidValue` → 插件内部断言失败；
- **不是显存（VRAM 6GB 够用），是共享内存（Shared Memory）**；**改代码无法绕过**（闭源插件 + 硬性 API 校验）。

三条出路：
1. **换 RTX 30 系卡**（sm_86，上限 100KB）→ 不用改任何代码，重转 engine 即可；
2. **换开源立体匹配**（SGBM/SGM，1660 SUPER 足够）→ 只改 `ess_depth_engine.cpp` 一个模块的 infer() 实现，对外接口不变；
3. **用 Jetson Orin**（sm_87，上限 100KB，ESS 原生目标平台）→ 见第 9 节。

### 8.2 已知问题：YUYV 模式在 Jetson Orin 软件更新后失效（实测，2026-08）

**现象**：Jetson Orin 上原本 YUYV 3840x1080@30 正常出帧；一次系统软件更新（L4T/内核/apt）后，YUYV 变成 `select() timeout` + `usb frame unavailable`，**把 fourcc 改成 MJPG 后恢复**。x86 台式机上也观测到类似超时（相机在总线上消失，疑似同一相机固件的连带症状）。

**为什么更新会影响"格式可用性"** —— YUYV 能否工作取决于整条 USB 链路，系统更新可能改到其中任何一层：

| 层 | 更新可能改变的东西 | 对 YUYV（249 MB/s 裸流）的影响 |
|---|---|---|
| 物理层 | USB PHY 固件/DTB 参数、电源管理 | 相机从 USB 3.0 协商**降级到 USB 2.0**（480M 有效仅 ~35 MB/s，249 MB/s 直接撑爆 → 必超时） |
| 协议层 | 内核升级 → uvcvideo/xHCI 驱动行为变化（带宽预算、帧间隔校验变严格；isochronous 调度） | 高带宽流被拒或丢帧 |
| 相机侧 | 相机固件本身不会因 PC 更新而变化，但它在高压下本来就不稳 | 持续大流量卡死 |

MJPG 不受影响的原因：硬件压缩后仅 ~30-60 MB/s，USB 2.0 都能跑，任何一层的小变化都压不垮它。

**验证命令**（在 Jetson 上跑，第一个最直接）：

```bash
lsusb -t                                  # ① 相机是 5000M 还是 480M？（降级则②③④）
uname -r                                  # ② 当前内核版本（对比更新前的）
dmesg | grep -iE "uvc|bandwidth|isoc|xhci" | tail -20   # ③ 带宽相关警告
ls /boot/ | grep vmlinuz                  # ④ 是否还保留旧内核（可回退）
```

**能否恢复**：
- 若是 USB 3.0 降级（lsusb 显示 480M）：换 USB 3.0 口/线、重新插拔、确认 Jetson 供电 → 大概率恢复；
- 若是驱动变严格：`apt` 里还有旧内核可回退试试，或升级到最新 JetPack（NVIDIA 已知修复了几个 xHCI 带宽问题）；
- **务实方案**：保持 MJPG。对下游零影响（ESS 输入还会缩到 480x288），不值得为 YUYV 折腾系统。

---

## 9. Jetson Orin 部署清单（如果选这条路）

Orin 是 ESS 的原生目标平台（插件自带 `aarch64/ess_plugins.so` 就是给它的），共享内存 100KB 满足要求。

1. 系统：JetPack 6.x（自带 TRT 10.x + CUDA 12.6，与插件编译版本匹配）+ Ubuntu 22.04 + ROS 2 Humble；
2. `params.yaml` 中 `ess.plugin_path` 改为 `src/wheel_perception/ess/plugins/aarch64/ess_plugins.so`；
3. **engine 文件必须重转**（engine 与 TRT 版本绑定，x86 上转的不能在 ARM 上用）：
   ```bash
   cd ~/wheel_ess/src/wheel_perception/ess
   # 把 build_all_trt10.sh 里的 TRT 路径改成 Orin 上的（/usr/lib/aarch64-linux-gnu 或 JetPack 实际路径）
   ./build_all_trt10.sh
   ```
4. 编译：CMakeLists.txt 已有 aarch64 分支（CUDA arch 自动 `72;87`），`colcon build` 即可；
5. `usb.device` 换成 Orin 上 `ls /dev/v4l/by-id/` 看到的相机路径；
6. 启动前确认 `source trt10_env.sh` 的路径在 Orin 上对应为 JetPack 的库路径。

---

## 10. 改动总览（一句话版）

> 换一个驱动（ZED SDK → USB V4L2）、加一个深度引擎（ESS TRT 插件）、
> 改一个参数文件（params.yaml 的 camera.type/usb/ess 段）、
> 在 fusion_node.cpp 的取帧处加一个分支——**下游感知与控制代码零改动**。

---

## 11. 跨平台移植清单（换新电脑：x86 或 ARM）

### 11.1 结论

换电脑**只需改 3 处配置 + 重转 2 样东西**，源码（fusion_node / 感知 / 控制）一行不用动：

| 动作 | 涉及文件 | 说明 |
|---|---|---|
| 改环境路径 | CMakeLists.txt | nvcc 路径、TRT_ROOT、CUDA_LIB_PATH（第 6 节） |
| 改库路径 | ess/trt10_env.sh | LD_LIBRARY_PATH 里的 TRT/CUDA 目录 |
| 改设备与插件路径 | config/params.yaml | `usb.device`（相机名变了）+ `ess.plugin_path`（架构变了） |
| **重转 engine** | ess/ 目录 | `build_all_trt10.sh`（在目标机上跑） |
| **重选插件 .so** | ess/plugins/ | x86_64 ↔ aarch64（不能拷，架构绑定） |

### 11.2 为什么 engine 和插件必须"重转/重选"而不能拷

```
engine（.engine 文件）     绑定：TRT 版本（10.x）+ GPU 架构（sm_75/86/87）+ 转换机器
插件（ess_plugins.so）    绑定：CPU 架构（x86_64 / aarch64）+ TRT 版本（≥10.3）
```

- **engine 拷过去基本必挂**：x86 转的 engine 不能在 ARM 上加载；TRT 10.9 转的在 TRT 10.3 机器上也可能加载失败。**规则：engine 在目标机器上用目标机的 TRT 重转**；
- **插件是编译期架构绑定**：`plugins/x86_64/ess_plugins.so` 不能用在 Jetson 上，反之亦然。x86 装 ARM 版会 dlopen 直接失败；
- 已验证的版本关系：插件按 TRT 10.3 编译 → 目标机 TRT **≥ 10.3**（10.9 实测 OK）；TRT 10.x 官方要求 **CUDA 12.x**（TRT 8.6 才是最后支持 CUDA 11 的版本）。

### 11.3 逐项检查清单

```bash
# ① 新机器上先核对环境（以下命令的输出决定了要改哪些路径）
nvcc --version                        # CUDA 编译工具链（哪一版、在哪个目录）
ldconfig -p | grep -i nvinfer         # TRT 运行库版本（libnvinfer.so.10 = TRT 10）
ls /dev/v4l/by-id/                    # USB 相机的实际设备名（改 params.yaml 用）
```

| # | 检查项 | x86 PC 示例（本机当前值） | Jetson Orin (ARM) 示例 |
|---|---|---|---|
| 1 | nvcc 路径（CMakeLists HINTS + `CUDACXX`） | `/usr/local/cuda-11.8/bin` | `/usr/local/cuda/bin`（JetPack 自带） |
| 2 | TRT 头文件+库（CMakeLists `TRT_ROOT`） | `/usr/local/cuda-12.2/targets/x86_64-linux` | `/usr/lib/aarch64-linux-gnu` 或 `/usr/local/tensorrt` |
| 3 | CUDA 运行库（CMakeLists `CUDA_LIB_PATH` + trt10_env.sh） | `/usr/local/cuda-11.8/lib64`（链接用） | `/usr/local/cuda/lib64` |
| 4 | trt10_env.sh 的 LD_LIBRARY_PATH | pip 的 nvidia/… + TRT_ROOT/lib | JetPack 的 tegra/… 库目录 |
| 5 | engine 重转（目标机上跑 build_all_trt10.sh） | 本机已转（TRT 10.9） | **必须重转**（JetPack 6.x 的 TRT 10.x） |
| 6 | 插件架构（params.yaml `ess.plugin_path`） | `plugins/x86_64/ess_plugins.so` | `plugins/aarch64/ess_plugins.so` |
| 7 | USB 设备名（params.yaml `usb.device`） | `/dev/v4l/by-id/usb-Generic_…` | 按 Orin 上 `ls /dev/v4l/by-id/` 填 |
| 8 | CUDA 架构（CMakeLists 自动选择，无需改） | `75;86`（x86 分支） | `72;87`（aarch64 分支，代码里已有） |

### 11.4 不需要动的东西

- **标定文件** `camera/stereo_calibration.json`——相机没换就不变；
- **全部源码**——fusion_node.cpp、感知/控制、launch 文件跨平台通用；
- **params.yaml 的相对路径**——所有 engine/标定路径都是相对 `~/wheel_ess` 工作目录的，只要保持目录结构不变就有效；
- CUDA 架构选择——CMakeLists.txt 已按 `CMAKE_SYSTEM_PROCESSOR` 自动分 x86/ARM，不用手改。

### 11.5 移植流程（照顺序做，5 分钟）

```bash
# 1. 新机器：装好 ROS 2 Humble + OpenCV + jsoncpp + Eigen，然后
nvcc --version && ldconfig -p | grep -i nvinfer     # 核对版本

# 2. 改 CMakeLists.txt 的三个路径（11.3 表 #1#2#3）
# 3. 改 ess/trt10_env.sh 的 LD_LIBRARY_PATH（表 #4）
# 4. 改 params.yaml：usb.device + ess.plugin_path（表 #6#7）

# 5. 重转 engine（在目标机上！）
source trt10_env.sh
cd ~/wheel_ess/src/wheel_perception/ess
./build_all_trt10.sh        # 若脚本里 TRT10_ROOT 写死，改成目标机路径

# 6. 编译 + 运行
cd ~/wheel_ess && colcon build --packages-select wheel_perception
source install/setup.bash && ros2 launch wheel_perception run_launch.py
```

> 一句话：**换电脑 = 改 3 个路径（CMake/trt10_env.sh/params.yaml）+ 目标机上重转 engine + 换对架构的插件**，源码零改动。

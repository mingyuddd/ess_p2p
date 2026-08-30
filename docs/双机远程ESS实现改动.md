# 双机远程 ESS 深度推理（实现改动说明）

> 本文档记录「双机远程 ESS 深度推理」功能相对原工程的**全部代码改动**：
> 改了哪些文件、加/改了哪些函数、为什么这么设计。
> 使用步骤见 [双机远程ESS深度推理.md](双机远程ESS深度推理.md)。

---

## 0. 改动总览

| # | 位置 | 改动类型 | 作用 |
|---|---|---|---|
| 1 | `src/wheel_msgs/msg/StereoCalibration.msg` | 新增消息 | 相机标定传输载体 |
| 2 | `src/wheel_msgs/CMakeLists.txt` | 注册消息 | 生成消息代码 |
| 3 | `usb_stereo_driver.{hpp,cpp}`（两份副本） | 新增接口 | 取左右目 CPU 校正图 |
| 4 | `src/wheel_perception/src/remote_ess_server.cpp` | 新增节点 | 服务端：收图推理回深度 |
| 5 | `src/wheel_perception/src/fusion_node.cpp` | 客户端改造 | ess_remote 模式链路 |
| 6 | `config/params.yaml` | 新增参数段 | 双机运行配置 |
| 7 | `launch/remote_ess_server.launch.py` | 新增 launch | 目标机启动服务端 |
| 8 | `CMakeLists.txt` | 注册新节点 | 编译/安装 remote_ess_server |

---

## 1. wheel_msgs：新增标定消息

**`src/wheel_msgs/msg/StereoCalibration.msg`**（新文件）：

```
std_msgs/Header header
float32 fx
float32 fy
float32 cx
float32 cy
float32 baseline_m
```

**`src/wheel_msgs/CMakeLists.txt`**：`rosidl_generate_interfaces` 中加入一行 `"msg/StereoCalibration.msg"`。

> 作用：客户端（本机）把相机标定传给服务端（目标机），服务端推理需要 fx/fy/cx/cy/baseline 才能把视差转成米。

## 2. 驱动接口：retrieveStereoPair

**声明**（两份副本保持镜像同步）：
- `src/wheel_perception/include/wheel_perception/core/usb_stereo_driver.hpp`
- `usb_ess_sdk/include/usb_ess_sdk/usb_stereo_driver.hpp`

**实现**：`usb_stereo_driver.cpp` 中新增：

```cpp
void UsbStereoDriver::retrieveStereoPair(cv::Mat& out_left_bgr, cv::Mat& out_right_bgr) const {
    impl_->left_rect_bgr.copyTo(out_left_bgr);
    impl_->right_rect_bgr.copyTo(out_right_bgr);
}
```

> 驱动在 `grab` 时已缓存左右目校正图（CPU BGR，1280x720），该接口直接取出，供客户端组帧发送——不需要再走 GPU 拷贝。

## 3. 服务端节点：remote_ess_server.cpp（新增，约 280 行）

组件模式，`RCLCPP_COMPONENTS_REGISTER_NODE(RemoteEssServer)`，部署在目标机。

**参数**（全部来自 params.yaml）：
- `ess.*`（复用现有段）：engine_path / plugin_path / input_width / input_height / confidence / min/max_depth
- `ess_remote.topics.*`：话题名
- `ess_remote.jpeg_quality`：JPEG 解码参考

**主流程 `onStereoImage`**（回调内同步串行推理）：

```
1. imdecode 解码拼接 JPEG → BGR 全幅
2. 左右各切一半（Rect(0,0,w,h) / Rect(w,0,w,h)）→ cvtColor 转 BGRA（引擎输入要求）
3. cudaMallocPitch 缓冲（按尺寸惰性分配、复用）→ cudaMemcpy2D 上传左右目
4. ess_engine_->infer(left, pitch, right, pitch, w, h, calibration, output)
5. waitForCompletion() 等待完成
6. 深度 D2H → float 米 → 16UC1 毫米（>0.1 且 <20m 有效，否则 0）
7. 发布 depth，header 原样带回输入时间戳
8. 每 60 帧打印 getTimingStats（pre/infer/depth/total + worst）
```

**标定订阅**：`rclcpp::QoS(1).transient_local()`（订阅端兼容 volatile 发布端），收到前跳过推理并 2 秒限频告警。

> 串行性由单线程 Executor + `keep_last(1)` 保证：回调阻塞期间新帧覆盖队列，始终处理最新帧。

## 4. 客户端改造：fusion_node.cpp（ess_remote 模式）

### 4.1 参数（on_configure）

```cpp
remote_stereo_topic_  = declare_parameter("ess_remote.topics.stereo_input", "ess_remote/stereo_input");
remote_depth_topic_   = declare_parameter("ess_remote.topics.depth", "ess_remote/depth");
remote_calib_topic_   = declare_parameter("ess_remote.topics.calibration", "ess_remote/calibration");
remote_publish_interval_ms_ = declare_parameter<int>("ess_remote.publish_interval_ms", 33); // ~30Hz
remote_jpeg_quality_  = declare_parameter<int>("ess_remote.jpeg_quality", 85);
remote_stale_timeout_ms_ = declare_parameter<int64_t>("ess_remote.stale_timeout_ms", 500);
```

### 4.2 init_camera（usb 分支）

- `depth_source_` 校验放宽为 `"ess" | "ess_remote"`
- `"ess_remote"` 分支：**不初始化 EssDepthEngine**（本机不跑模型），改为创建：
  - `pub_remote_stereo_`：LifecyclePublisher<sensor_msgs::msg::Image>，`SensorDataQoS().keep_last(1)`（best_effort，发拼接图）
  - `pub_remote_calib_`：LifecyclePublisher<StereoCalibration>，`rclcpp::QoS(1)`（**volatile**，见 §6 坑）
  - `sub_remote_depth_`：订阅远程深度 → `remote_depth_callback`（只锁存指针，复用 `frame_mutex_`）

### 4.3 run_ess_depth 分流

```cpp
bool run_ess_depth(core::ZedGpuFrame& frame) {
    if (depth_source_ == "ess_remote") {
        return run_ess_depth_remote(frame);   // 本地不推理，走远程链路
    }
    ...
}
```

### 4.4 run_ess_depth_remote（新增，核心逻辑）

**① 节流发图**（距上次发布 ≥ publish_interval_ms）：

```cpp
usb_driver_->retrieveStereoPair(left_bgr, right_bgr);
// 标定：随每帧携带（volatile 下不依赖 latched）
const auto p = usb_driver_->getCameraParams();  // fx/fy/cx/cy/baseline
pub_remote_calib_->publish(calib_msg);
// 左右拼接 → JPEG
cv::hconcat(left_bgr, right_bgr, stereo);
cv::imencode(".jpg", stereo, jpg, {cv::IMWRITE_JPEG_QUALITY, remote_jpeg_quality_});
pub_remote_stereo_->publish(img_msg);  // stamp = frame.timestamp_ns
```

**② 取最新远程深度 + 新鲜度判断**：

```cpp
depth_msg = latest_remote_depth_;                    // mutex 保护
if (!depth_msg)                      → 限频告警，cloud 置空，返回 true
age_ms = (now - depth.header.stamp)  → 超 stale_timeout_ms(500) 视为过期
                                     → 置空 cloud，返回 true（跳过点云融合，AI 照常）
```

**③ 本地重建点云**（复刻 dataset 模式 BLOCK 1，坐标语义 Z_UP_X_FWD）：

```cpp
// 16UC1 毫米 → float 米 → float4（X前=d, Y左=d*(cx-u)/fx, Z上=d*(cy-v)/fy）
// cudaMemcpy 上传 d_cloud_buffer_ → frame.cloud_ptr_dev / cloud_step_bytes
```

> 返回 `true` 但 `cloud_ptr_dev=nullptr` 时，下游 `has_depth_input=false` 走既有安全分支
> （跳过 filterCloud/extractRoadEdges，AI 语义分割照常），**帧不丢**。

### 4.5 生命周期与成员

- `on_activate` / `on_deactivate`：远程两个发布者 on_activate / on_deactivate
- `on_cleanup`：reset 远程发布者/订阅者/`latest_remote_depth_`，清零节流时间戳
- 新成员：话题名、interval/jpeg_quality/stale_timeout、`pub_remote_stereo_`、`pub_remote_calib_`、`sub_remote_depth_`、`latest_remote_depth_`、`remote_calib_published_`、`last_remote_stereo_pub_ns_`

## 5. 配置与 launch

**`config/params.yaml`** 新增：

```yaml
ess_remote:
  topics:
    stereo_input: "ess_remote/stereo_input"
    depth: "ess_remote/depth"
    calibration: "ess_remote/calibration"
  publish_interval_ms: 33
  jpeg_quality: 85
  stale_timeout_ms: 500
```

`zed.depth_source` 注释更新为 `sdk | ess | ess_remote`。客户端 launch（run_launch.py 等）**无需改动**，切参数即可。

**`launch/remote_ess_server.launch.py`**（新增，目标机用）：ComposableNodeContainer 加载 `RemoteEssServer` 组件，加载同一份 params.yaml（复用 `ess` 段），无 lifecycle 命令。

## 6. CMakeLists 与关键设计决策

**CMakeLists.txt**：

```cmake
add_library(remote_ess_server SHARED src/remote_ess_server.cpp)
target_link_libraries(remote_ess_server wheel_core ${OpenCV_LIBS})
ament_target_dependencies(remote_ess_server rclcpp rclcpp_components sensor_msgs std_msgs cv_bridge wheel_msgs)
rclcpp_components_register_nodes(remote_ess_server "RemoteEssServer")
# install(TARGETS) 列表加 remote_ess_server
```

**关键决策与坑**（按重要度）：

| 决策 | 原因 |
|---|---|
| **话题流水线而非 service** | ESS 30Hz 实时流，service 只适合低频请求（见 STUDY/0_0ROS2应用.md §2.5 vs §3.6） |
| **拼接单张 JPEG 传图** | 避免 image_transport 及其 Lifecycle 兼容问题；单话题保证左右目配对不撕裂 |
| **深度 raw 不压缩** | JPEG 会损坏深度精度；16UC1 毫米与本地模式编码一致 |
| **标定 volatile + 每帧携带**（踩坑后改） | 工程 launch 开 `use_intra_process_comms=True`，**进程内通信不允许 latched**，`transient_local()` 会在 configure 抛 `intraprocess communication allowed only with volatile durability`；改为每帧携带后服务端晚启动也能收到 |
| **客户端加载需 TRT 库可解析** | fusion_node 链接的 `libwheel_core.so` 编译了 ESS 代码，动态链接器要求 `libnvinfer.so.10` 存在（找到 ≠ 用它推理）。本机目录 `/usr/local/cuda-12.2/targets/x86_64-linux/lib` 有库，`export LD_LIBRARY_PATH=...` 即可 |
| **服务端串行推理** | 单线程 Executor + keep_last(1)，积压时自动丢旧帧，始终处理最新 |
| **时间戳链路** | frame.timestamp_ns → 图 header.stamp → 服务端原样带回 → 客户端 500ms 新鲜度判断 |

## 7. 验证记录

- ✅ 本机 `colcon build --packages-select wheel_msgs wheel_perception` 通过
- ✅ 组件加载：容器内 `RemoteEssServer` 成功注册，三个话题出现
- ✅ 真实 launch（intra-process 开启）：configure 通过，`Transitioning successful`，客户端按 30Hz 发图 + 每帧携带标定
- ✅ 退化行为：服务端未连时客户端限频告警 `no remote depth yet`，不崩溃、帧不丢
- ⏳ 双机联调（需实机）：`ros2 topic hz /ess_remote/depth` ≈ 30Hz，rviz 点云正常

## 8. 文件清单（改动/新增）

| 状态 | 文件 |
|---|---|
| 新增 | `src/wheel_msgs/msg/StereoCalibration.msg` |
| 新增 | `src/wheel_perception/src/remote_ess_server.cpp` |
| 新增 | `src/wheel_perception/launch/remote_ess_server.launch.py` |
| 修改 | `src/wheel_msgs/CMakeLists.txt` |
| 修改 | `src/wheel_perception/include/wheel_perception/core/usb_stereo_driver.hpp` |
| 修改 | `src/wheel_perception/src/core/usb/usb_stereo_driver.cpp` |
| 修改 | `usb_ess_sdk/include/usb_ess_sdk/usb_stereo_driver.hpp` |
| 修改 | `usb_ess_sdk/src/usb_stereo_driver.cpp` |
| 修改 | `src/wheel_perception/src/fusion_node.cpp` |
| 修改 | `src/wheel_perception/config/params.yaml` |
| 修改 | `src/wheel_perception/CMakeLists.txt` |

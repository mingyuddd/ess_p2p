# 双机远程 ESS 深度推理（使用文档）

> 适用场景：本机（客户端）无法运行 ESS 深度模型（如 GPU 算力不足/环境不匹配），
> 但目标电脑已有完整 wheel_ess 工程环境（CUDA + TensorRT + 引擎文件齐全）可以跑模型。
> 方案：本机采集 USB 双目 → 发图给目标机 → 目标机跑 ESS 推理 → 深度回传 →
> 本机重建点云接入原有融合链路。**模型推理只在目标机发生，本机只发图收深度。**

---

## 1. 架构与数据流

```
本机 (客户端 fusion_node, zed.depth_source=ess_remote)        目标机 (服务端 remote_ess_server)
┌────────────────────────────────────────┐          ┌────────────────────────────────────────┐
│ USB 双目 grab → retrieveStereoPair     │          │ 订阅 ess_remote/stereo_input           │
│ 左右拼接 2560x720 → JPEG 压缩 (85)     │   JPEG   │ imdecode → 切左右 → BGR→BGRA            │
│ ──发布──▶ ess_remote/stereo_input ─────┼─────────▶│ cudaMemcpy2D 上传 GPU                   │
│ (best_effort, keep_last(1))            │          │ EssDepthEngine::infer (GPU 推理)        │
│                                        │          │ waitForCompletion → 深度 D2H            │
│ 订阅 ess_remote/depth (16UC1 毫米)     │◀─────────┼── 发布 ── ess_remote/depth              │
│ 深度 → CPU 重建 float4 点云 → 上传 GPU  │  深度    │ (best_effort, keep_last(1))             │
│ → 原有 filterCloud / 道路边缘 链路照常  │          │                                        │
│                                        │          │ 订阅 ess_remote/calibration            │
│ ──发布──▶ ess_remote/calibration ──────┼─────────▶│ (transient_local, 收到后才开始推理)    │
│ (latched, 启动时发布一次)              │          │                                        │
└────────────────────────────────────────┘          └────────────────────────────────────────┘
``` 

关键设计：

- **单话题传图**：左右目拼接成单张 2560x720 再 JPEG 压缩（`cv::imencode`），保证左右配对、避免 image_transport 依赖
- **深度不压缩**：16UC1 毫米编码 raw 传输，JPEG 会损失深度精度
- **标定动态下发**：客户端**随每帧**携带相机标定（volatile，仅 5 个 float 开销可忽略；工程 launch 开了 intra-process 不能用 latched），服务端收到后才开始推理，晚启动也能在下一帧收到——换相机无需改服务端
- **时间戳链路**：相机帧时间戳 → 图像 `header.stamp` → 服务端深度 `header.stamp` 原样带回 → 客户端以此判断深度新鲜度（默认 500ms）

## 2. 消息契约

| 话题 | 类型 | QoS | 内容 |
|---|---|---|---|
| `ess_remote/stereo_input` | sensor_msgs/Image | best_effort + keep_last(1) | 左右拼接 JPEG（encoding=`"jpeg"`），2560x720 BGR |
| `ess_remote/depth` | sensor_msgs/Image | best_effort + keep_last(1) | 深度图 16UC1 毫米，1280x720（encoding=`"16UC1"`） |
| `ess_remote/calibration` | wheel_msgs/msg/StereoCalibration | volatile（可靠，随每帧携带） | fx/fy/cx/cy/baseline_m，客户端每帧发布（服务端晚启动也能收到） |

`StereoCalibration.msg`：

```
std_msgs/Header header
float32 fx
float32 fy
float32 cx
float32 cy
float32 baseline_m
```

## 3. 部署前提

| 机器 | 角色 | 要求 |
|---|---|---|
| 目标机 | 服务端 | 完整 wheel_ess 工程环境（能跑通原工程即可：CUDA + TensorRT + 引擎文件） |
| 本机 | 客户端 | 能采集 USB 双目；无需 ESS 环境（见 §5.1 加载说明） |

两台机器需满足：

- 同一局域网（深度下行约 420 Mbps @30Hz，千兆网推荐）
- `ROS_DOMAIN_ID` 相同（默认 0 即可，两台都未设置时相同）
- 同源代码（同一份 wheel_ess 工程），避免消息定义不一致

## 4. 目标机部署（服务端）

```bash
# 1. 编译（新增节点 remote_ess_server 与消息 StereoCalibration）
colcon build --packages-select wheel_msgs wheel_perception

# 2. 启动服务端（复用 params.yaml 的 ess 段：engine_path / plugin_path 等）
source install/setup.bash
ros2 launch wheel_perception remote_ess_server.launch.py
```

启动后应看到：

```
[remote_ess_server]: ESS engine ready: input 480x288, output 1280x720
[remote_ess_server]: RemoteESS server up: sub ess_remote/stereo_input (JPEG stereo) | pub ess_remote/depth (16UC1 depth), waiting for calibration on ess_remote/calibration
```

收到客户端标定后：

```
[remote_ess_server]: Calibration received: fx=... fy=... cx=... cy=... baseline=... m
```

## 5. 本机部署（客户端）

### 5.1 加载说明（重要）

fusion_node 链接的共享库 `libwheel_core.so` 里编译了 ESS 引擎代码（含 TensorRT），
动态链接器加载 fusion_node 时需要能解析到 TRT 运行库——**这与本机跑不跑模型无关**。
如果启动时报 `libnvinfer.so.10: cannot open shared object file`，在启动前执行：

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

（路径以本机编译 CMakeLists 中 `TRT_ROOT` 实际路径为准；建议写进 `~/.bashrc`）

### 5.2 配置

编辑 `src/wheel_perception/config/params.yaml`：

```yaml
zed:
  depth_source: "ess_remote"   # 原来是 "ess"，改为 "ess_remote"
```

### 5.3 启动

```bash
source install/setup.bash
ros2 launch wheel_perception usb_ess_perception_only.launch.py   # 工程原有 launch，无需改动
```

## 6. 联调验证

在客户端机器上：

```bash
# 1. 三个话题应全部出现
ros2 topic list | grep ess_remote

# 2. 深度频率应与相机帧率一致（约 30Hz）
ros2 topic hz /ess_remote/depth

# 3. 检查深度内容（非 0 值应为真实距离毫米数）
ros2 topic echo /ess_remote/depth --once | head

# 4. 服务端日志每 60 帧打印一次推理耗时
#    [60 frames] pre ...ms infer ...ms depth ...ms total ...ms (worst ...ms)

# 5. rviz 查看 zed/point_cloud 点云正常
```

服务端日志中 `infer` 耗时即模型推理时间（目标机 GPU 决定）；客户端可用
`ros2 topic delay /ess_remote/depth` 观察往返延迟（含网络，正常几十 ms 量级）。

## 7. 参数说明

### 7.1 客户端 `ess_remote` 段（params.yaml）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `ess_remote.topics.stereo_input` | `ess_remote/stereo_input` | 拼接图发送话题 |
| `ess_remote.topics.depth` | `ess_remote/depth` | 深度接收话题 |
| `ess_remote.topics.calibration` | `ess_remote/calibration` | 标定发布话题 |
| `ess_remote.publish_interval_ms` | 33 | 发图节流（约 30Hz） |
| `ess_remote.jpeg_quality` | 85 | JPEG 压缩质量 0-100（越低带宽越小、画质越差） |
| `ess_remote.stale_timeout_ms` | 500 | 深度超过该时长未回传视为过期 |

### 7.2 服务端 `ess` 段（复用）

目标机引擎配置沿用 params.yaml 现有 `ess` 段：`engine_path`、`plugin_path`、
`input_width`、`input_height`、`confidence_threshold`、`min_depth_m`、`max_depth_m`。
无需单独修改——服务端 launch 加载的就是同一份 params.yaml。

## 8. 行为与故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 本机启动报 `libnvinfer.so.10` 找不到 | TRT 运行库未进 ldconfig | §5.1 的 `export LD_LIBRARY_PATH` |
| 服务端日志反复"Waiting for calibration" | 客户端还没启动，或标定话题没收到 | 客户端启动后（每帧携带标定）自动收到，无需重启任何一方 |
| 客户端日志"no remote depth yet" | 服务端未启动或网络不通 | 检查服务端进程、`ros2 node list` 两机互通、`ROS_DOMAIN_ID` |
| 客户端日志"remote depth stale" | 深度回传超过 500ms | 检查网络质量/服务端推理耗时；可调大 `stale_timeout_ms` |
| 客户端日志"depth format mismatch" | 版本不一致 | 两机同步代码重新编译 |
| 杀掉服务端 | —— | 客户端不崩溃，自动降级：跳过点云融合（避障失效），AI 语义分割照常；服务端恢复后自动续上 |

## 9. 注意事项

- **延迟**：远程推理引入约 30-80ms 额外延迟（网络 + 串行链路），实车测试时注意避障反应距离
- **带宽**：深度 16UC1 1280x720 @30Hz ≈ 420 Mbps 下行，千兆网内可行；网络差时可降低 `usb.fps` 或调低 `jpeg_quality`（只影响上行图，深度不压缩）
- **精度**：16UC1 毫米编码与本地 ESS 模式精度一致，点云坐标语义不变（X前 Y左 Z上）
- **服务端为串行推理**：单线程 Executor 逐帧处理，若网络抖动导致积压会自动丢弃旧帧（keep_last(1)），始终处理最新一帧

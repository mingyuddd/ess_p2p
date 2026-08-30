// =========================================================
// remote_ess_server: 远程 ESS 深度推理服务端节点（双机模式）
//
// 数据流：
//   本机（客户端, fusion_node depth_source=ess_remote）把 USB 双目
//   左右拼接图经 JPEG 压缩发到 ess_remote/stereo_input；
//   本节点 imdecode 解码 -> 切分左右目 -> BGR->BGRA ->
//   cudaMemcpy2D 上传 GPU -> EssDepthEngine::infer ->
//   waitForCompletion -> 深度 D2H -> 16UC1 毫米 ->
//   发回 ess_remote/depth（header.stamp 与输入图一致，
//   客户端据此判断新鲜度）。
//
// 标定：客户端随每帧经 ess_remote/calibration 携带（volatile 周期发送，
//   本节点订阅端必须用 volatile 匹配，勿用 transient_local——DDS 兼容性规则下
//   transient_local 订阅收不到 volatile 发布者的数据），收到后才开始推理——
//   换相机无需改服务端。
//
// 部署在能跑 ESS 模型的目标机器上（完整工程环境）。组件模式，
// 由 launch/remote_ess_server.launch.py 加载。
// =========================================================

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "wheel_msgs/msg/stereo_calibration.hpp"
#include "wheel_perception/core/ess_depth_engine.hpp"

class RemoteEssServer : public rclcpp::Node {
public:
    explicit RemoteEssServer(const rclcpp::NodeOptions& options);
    ~RemoteEssServer() override;

private:
    void onStereoImage(const sensor_msgs::msg::Image::SharedPtr msg);
    void onCalibration(const wheel_msgs::msg::StereoCalibration::SharedPtr msg);
    bool ensureUploadBuffers(int width, int height);
    void freeUploadBuffers();

    // ---- 参数 ----
    std::string stereo_topic_;
    std::string depth_topic_;
    std::string calibration_topic_;
    int jpeg_quality_;

    // ---- ESS 引擎（复用 params.yaml 现有 ess 段，与 fusion_node 本地模式一致）----
    std::unique_ptr<wheel_perception::core::EssDepthEngine> engine_;

    // ---- 标定（客户端每帧携带，收到前不推理）----
    wheel_perception::core::EssDepthEngine::StereoCalibration calibration_;
    bool have_calibration_ = false;

    // ---- 话题 ----
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_depth_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_stereo_;
    rclcpp::Subscription<wheel_msgs::msg::StereoCalibration>::SharedPtr sub_calibration_;

    // ---- GPU 上传缓冲（按图像尺寸惰性分配、复用）----
    void* d_left_ = nullptr;
    void* d_right_ = nullptr;
    size_t d_pitch_ = 0;
    int buffer_width_ = 0;
    int buffer_height_ = 0;

    // ---- 深度输出缓冲（复用）----
    std::vector<float> depth_f32_host_;
    std::vector<uint16_t> depth_mm_host_;

    // ---- 统计 ----
    uint64_t frame_count_ = 0;
    double worst_total_ms_ = 0.0;
    bool depth_published_ = false;  // 首个深度帧发布后打印一次，确认全链路 OK
};

RemoteEssServer::RemoteEssServer(const rclcpp::NodeOptions& options)
    : Node("remote_ess_server", options) {
    // ---- 推理引擎参数（复用现有 ess 段）----
    const std::string engine_path = declare_parameter<std::string>(
        "ess.engine_path", "src/wheel_perception/ess/light_ess.engine");
    const std::string plugin_path = declare_parameter<std::string>(
        "ess.plugin_path", "src/wheel_perception/ess/plugins/x86_64/ess_plugins.so");
    wheel_perception::core::EssDepthEngine::Config ess_cfg;
    ess_cfg.engine_path = engine_path;
    ess_cfg.plugin_path = plugin_path;
    ess_cfg.input_width = declare_parameter<int>("ess.input_width", 480);
    ess_cfg.input_height = declare_parameter<int>("ess.input_height", 288);
    ess_cfg.confidence_threshold = declare_parameter<double>("ess.confidence_threshold", 0.5);
    ess_cfg.min_depth_m = declare_parameter<double>("ess.min_depth_m", 0.3);
    ess_cfg.max_depth_m = declare_parameter<double>("ess.max_depth_m", 10.0);

    // ---- 话题参数 ----
    stereo_topic_ = declare_parameter<std::string>(
        "ess_remote.topics.stereo_input", "ess_remote/stereo_input");
    depth_topic_ = declare_parameter<std::string>(
        "ess_remote.topics.depth", "ess_remote/depth");
    calibration_topic_ = declare_parameter<std::string>(
        "ess_remote.topics.calibration", "ess_remote/calibration");
    jpeg_quality_ = declare_parameter<int>("ess_remote.jpeg_quality", 85);

    // ---- 初始化引擎 ----
    engine_ = std::make_unique<wheel_perception::core::EssDepthEngine>(ess_cfg);
    if (!engine_->initialize()) {
        RCLCPP_ERROR(get_logger(),
                     "ESS engine init failed (engine=%s plugin=%s); node will idle.",
                     ess_cfg.engine_path.c_str(), ess_cfg.plugin_path.c_str());
    } else {
        RCLCPP_INFO(get_logger(), "ESS engine ready: input %dx%d, output %dx%d",
                    engine_->inputWidth(), engine_->inputHeight(),
                    engine_->outputWidth(), engine_->outputHeight());
    }

    // ---- 话题 ----
    pub_depth_ = create_publisher<sensor_msgs::msg::Image>(
        depth_topic_, rclcpp::SensorDataQoS().keep_last(1));

    // 必须 volatile（默认），与客户端 pub_remote_calib_ 的 rclcpp::QoS(1) 匹配；
    // transient_local 订阅会因 durability 不兼容而收不到 volatile 发布者的数据。
    sub_calibration_ = create_subscription<wheel_msgs::msg::StereoCalibration>(
        calibration_topic_, rclcpp::QoS(1),
        [this](const wheel_msgs::msg::StereoCalibration::SharedPtr msg) { onCalibration(msg); });

    sub_stereo_ = create_subscription<sensor_msgs::msg::Image>(
        stereo_topic_, rclcpp::SensorDataQoS().keep_last(1),
        [this](const sensor_msgs::msg::Image::SharedPtr msg) { onStereoImage(msg); });

    RCLCPP_INFO(get_logger(),
                "RemoteESS server up: sub %s (JPEG stereo) | pub %s (16UC1 depth), "
                "waiting for calibration on %s",
                stereo_topic_.c_str(), depth_topic_.c_str(), calibration_topic_.c_str());
}

RemoteEssServer::~RemoteEssServer() { freeUploadBuffers(); }

void RemoteEssServer::onCalibration(
    const wheel_msgs::msg::StereoCalibration::SharedPtr msg) {
    calibration_.fx = msg->fx;
    calibration_.fy = msg->fy;
    calibration_.cx = msg->cx;
    calibration_.cy = msg->cy;
    calibration_.baseline_m = msg->baseline_m;
    have_calibration_ = true;
    RCLCPP_INFO(get_logger(),
                "Calibration received: fx=%.2f fy=%.2f cx=%.2f cy=%.2f baseline=%.4f m",
                calibration_.fx, calibration_.fy, calibration_.cx, calibration_.cy,
                calibration_.baseline_m);
}

bool RemoteEssServer::ensureUploadBuffers(int width, int height) {
    if (d_left_ && width == buffer_width_ && height == buffer_height_) {
        return true;
    }
    freeUploadBuffers();
    if (cudaMallocPitch(&d_left_, &d_pitch_, static_cast<size_t>(width) * 4U,
                        static_cast<size_t>(height)) != cudaSuccess ||
        cudaMallocPitch(&d_right_, &d_pitch_, static_cast<size_t>(width) * 4U,
                        static_cast<size_t>(height)) != cudaSuccess) {
        RCLCPP_ERROR(get_logger(), "cudaMallocPitch failed for %dx%d upload buffers",
                     width, height);
        freeUploadBuffers();
        return false;
    }
    buffer_width_ = width;
    buffer_height_ = height;
    return true;
}

void RemoteEssServer::freeUploadBuffers() {
    if (d_left_) {
        cudaFree(d_left_);
        d_left_ = nullptr;
    }
    if (d_right_) {
        cudaFree(d_right_);
        d_right_ = nullptr;
    }
    d_pitch_ = 0;
    buffer_width_ = 0;
    buffer_height_ = 0;
}

void RemoteEssServer::onStereoImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "[step 0/7] stereo frame rcvd: stamp=%lld size=%zuB",
        static_cast<long long>(msg->header.stamp.sec) * 1000000000LL +
            static_cast<long long>(msg->header.stamp.nanosec),
        msg->data.size());

    if (!engine_ || !engine_->isInitialized()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
                             "ESS engine not initialized; skipping frames");
        return;
    }
    if (!have_calibration_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Waiting for calibration on '%s' (client publishes it "
                             "with every stereo frame)",
                             calibration_topic_.c_str());
        return;
    }

    // 1. 解码拼接 JPEG -> BGR 全幅（左 | 右）
    cv::Mat full = cv::imdecode(
        cv::Mat(1, static_cast<int>(msg->data.size()), CV_8UC1,
                const_cast<uint8_t*>(msg->data.data())),
        cv::IMREAD_COLOR);
    if (full.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "JPEG decode failed");
        return;
    }
    const int w = full.cols / 2;
    const int h = full.rows;
    if (w <= 0 || h <= 0) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Bad stereo image size %dx%d", full.cols, full.rows);
        return;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "[step 1/7] JPEG decoded: full %dx%d -> L/R %dx%d",
                         full.cols, full.rows, w, h);

    // 2. 切左右目 -> BGRA（引擎输入要求）
    cv::Mat left_bgra, right_bgra;
    cv::cvtColor(full(cv::Rect(0, 0, w, h)), left_bgra, cv::COLOR_BGR2BGRA);
    cv::cvtColor(full(cv::Rect(w, 0, w, h)), right_bgra, cv::COLOR_BGR2BGRA);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "[step 2/7] split to BGRA L/R done");

    // 3. 上传 GPU（同步拷贝；完成后 infer 在引擎 stream 上串行执行）
    if (!ensureUploadBuffers(w, h)) {
        return;
    }
    const size_t row_bytes = static_cast<size_t>(w) * 4U;
    const cudaError_t e1 = cudaMemcpy2D(
        d_left_, d_pitch_, left_bgra.data, left_bgra.step,
        row_bytes, static_cast<size_t>(h), cudaMemcpyHostToDevice);
    const cudaError_t e2 = cudaMemcpy2D(
        d_right_, d_pitch_, right_bgra.data, right_bgra.step,
        row_bytes, static_cast<size_t>(h), cudaMemcpyHostToDevice);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "cudaMemcpy2D upload failed: %s / %s",
                             cudaGetErrorString(e1), cudaGetErrorString(e2));
        return;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "[step 3/7] H2D upload ok (pitch=%zu)", d_pitch_);

    // 4. 推理（异步提交 -> 等待完成，串行处理每帧）
    wheel_perception::core::EssDepthEngine::InferenceOutput output;
    if (!engine_->infer(d_left_, d_pitch_, d_right_, d_pitch_, w, h,
                        calibration_, output)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "ESS inference failed; dropping frame");
        return;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "[step 4/7] inference submitted");
    if (!engine_->waitForCompletion()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "ESS inference sync failed; dropping frame");
        return;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "[step 5/7] inference completed");
    if (!output.depth_ptr_dev || output.width != w || output.height != h) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Unexpected ESS output size (%dx%d, want %dx%d)",
                             output.width, output.height, w, h);
        return;
    }

    // 6. 深度 D2H -> 16UC1 毫米（与客户端 BLOCK 1 dataset 编码一致）
    const size_t pixel_count = static_cast<size_t>(w) * static_cast<size_t>(h);
    depth_f32_host_.resize(pixel_count);
    depth_mm_host_.resize(pixel_count);
    if (cudaMemcpy(depth_f32_host_.data(), output.depth_ptr_dev,
                   pixel_count * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "Depth D2H copy failed");
        return;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "[step 6/7] depth D2H done (%zu px)", pixel_count);
    for (size_t i = 0; i < pixel_count; ++i) {
        const float d = depth_f32_host_[i];
        depth_mm_host_[i] =
            (d > 0.1F && d < 20.0F && std::isfinite(d))
                ? static_cast<uint16_t>(d * 1000.0F)
                : 0U;
    }

    // 7. 发布（header 原样带回输入时间戳，客户端据此判断新鲜度）
    auto depth_msg = std::make_unique<sensor_msgs::msg::Image>();
    depth_msg->header = msg->header;
    depth_msg->height = static_cast<uint32_t>(h);
    depth_msg->width = static_cast<uint32_t>(w);
    depth_msg->encoding = "16UC1";
    depth_msg->is_bigendian = false;
    depth_msg->step = static_cast<uint32_t>(w) * 2U;
    const uint8_t* depth_mm_ptr = reinterpret_cast<const uint8_t*>(depth_mm_host_.data());
    depth_msg->data.assign(depth_mm_ptr, depth_mm_ptr + depth_mm_host_.size() * sizeof(uint16_t));
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "[step 7a/7] about to publish depth (%dx%d 16UC1)", w, h);
    try {
        pub_depth_->publish(std::move(depth_msg));
    } catch (const std::exception& e) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                              "depth publish threw: %s", e.what());
        return;
    }
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "[step 7b/7] publish returned OK");
    if (!depth_published_) {
        depth_published_ = true;
        RCLCPP_INFO(get_logger(), "[ess_remote] first depth frame published; "
                                  "end-to-end pipeline OK");
    }

    // 8. 统计（每 60 帧打印一次引擎耗时）
    ++frame_count_;
    if (frame_count_ % 60 == 0) {
        wheel_perception::core::EssDepthEngine::TimingStats ts{};
        engine_->getTimingStats(ts);
        const double total = static_cast<double>(ts.total_ms);
        worst_total_ms_ = std::max(worst_total_ms_, total);
        RCLCPP_INFO(get_logger(),
                    "[%llu frames] pre %5.2fms infer %5.2fms depth %5.2fms total %5.2fms "
                    "(worst %.2fms)",
                    static_cast<unsigned long long>(frame_count_),
                    static_cast<double>(ts.preprocess_ms),
                    static_cast<double>(ts.inference_ms),
                    static_cast<double>(ts.depth_cloud_ms), total, worst_total_ms_);
    }
}

RCLCPP_COMPONENTS_REGISTER_NODE(RemoteEssServer)

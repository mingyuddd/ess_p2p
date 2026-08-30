#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
// [关键修复] 必须包含这个头文件才能用 Modifier 和 Iterator
#include <sensor_msgs/point_cloud2_iterator.hpp> 
#include <sensor_msgs/msg/imu.hpp> // [新增]
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp> // [ess_remote] cv::imencode JPEG 压缩

#include "wheel_msgs/msg/perception_output.hpp"
#include "wheel_msgs/msg/obstacle_info.hpp"
#include "wheel_msgs/msg/stereo_calibration.hpp" // [ess_remote] 远程标定消息
#include "wheel_perception/core/zed_driver.hpp"
#include "wheel_perception/core/usb_stereo_driver.hpp"
#include "wheel_perception/core/ess_depth_engine.hpp"
#include "wheel_perception/core/ai_engine.hpp"
#include "wheel_perception/core/obstacle_fusion.hpp"
#include <nav_msgs/msg/odometry.hpp> // [新增] 里程计消息头文件
#include <cuda_runtime.h>            // [新增] 必须包含 CUDA API 以便管理显存
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "wheel_perception/core/zed_driver.hpp"


using namespace wheel_perception;
using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

// 简单的辅助结构体
struct SceneMetrics {
    // 障碍物相关
    float min_dist = 1000.0f;
    geometry_msgs::msg::Point min_point; // ROS Body Frame (x前, y左)

    // 巡线相关
    bool has_line = false;
    float road_dist = 0.0f;      // 对应 msg.right_distance
    float road_yaw_rad = 0.0f;   // 对应 msg.road_yaw_error (弧度)
    bool has_road_width = false;
    float left_distance = 0.0f;
    float road_width = 0.0f;
    float target_right_distance = 0.0f;
    
    // 新增：表示路线是否接近 90 度 / 垂直
    bool is_vertical = false; 

    // 调试可视化用
    geometry_msgs::msg::Point debug_pt1; // 远端重心
    geometry_msgs::msg::Point debug_pt2; // 近端重心
};

class FusionNode : public rclcpp_lifecycle::LifecycleNode {
public:
    explicit FusionNode(const rclcpp::NodeOptions & options)
        : rclcpp_lifecycle::LifecycleNode("fusion_node", options) {}

    ~FusionNode() {
    if (timer_) timer_->cancel();
    if (driver_) {
        RCLCPP_INFO(get_logger(), "Ctrl+C detected, shutting down ZED camera...");
        driver_->close();
        driver_.reset();
    }
    if (usb_driver_) {
        RCLCPP_INFO(get_logger(), "Ctrl+C detected, shutting down USB stereo camera...");
        usb_driver_->close();
        usb_driver_.reset();
    }
    }
    core::PerceptionParams params_;
    bool dynamic_aim_enabled_ = true;
    float dynamic_aim_ratio_ = 0.35f;
    float dynamic_aim_min_dist_ = 0.65f;
    float dynamic_aim_max_dist_ = 1.0f;
    float dynamic_aim_fallback_dist_ = 1.0f;
    float dynamic_aim_min_width_ = 1.0f;
    float dynamic_aim_max_width_ = 15.0f;
    float dynamic_aim_max_target_step_ = 0.05f;
    int dynamic_aim_hold_frames_ = 10;

CallbackReturn on_configure(const rclcpp_lifecycle::State &) override {
    RCLCPP_INFO(get_logger(), "Configuring parameters...");
    // ================= 0. 运行模式选择 =================
    use_dataset_mode_ = this->declare_parameter<bool>("zed.use_dataset_mode", true);
    camera_type_ = this->declare_parameter<std::string>("camera.type", "zed");
    depth_source_ = this->declare_parameter<std::string>("zed.depth_source", "sdk");

    // ================= 1. AI 引擎参数 =================
    std::string engine_path = this->declare_parameter<std::string>("ai.engine_path", "src/wheel_perception/config/model_final_v2_combine.engine");

    // ================= 2. ZED 相机参数 (软编码化) =================
    core::ZedDriver::Config zed_cfg;
    zed_cfg.resolution_id = this->declare_parameter<int>("zed.resolution_id", 2); // 2=HD720
    zed_cfg.fps = this->declare_parameter<int>("zed.fps", 60);
    zed_cfg.depth_min = this->declare_parameter<float>("zed.depth_min", 0.3f);
    zed_cfg.depth_max = this->declare_parameter<float>("zed.depth_max", 5.0f);
    zed_cfg.flip_camera = this->declare_parameter<bool>("zed.flip_camera", false);
    zed_cfg.depth_mode = this->declare_parameter<int>("zed.depth_mode", 1); // PERFORMANCE
    zed_cfg.coordinate_system = this->declare_parameter<int>("zed.coordinate_system", 3); // Z_UP_X_FWD
    zed_cfg.svo_path = this->declare_parameter<std::string>("zed.svo_path", "");

    core::UsbStereoDriver::Config usb_cfg;
    usb_cfg.device = this->declare_parameter<std::string>(
        "usb.device", "/dev/v4l/by-id/usb-Generic_USB_Camera_YHTek-video-index0");
    usb_cfg.fourcc = this->declare_parameter<std::string>("usb.fourcc", "YUYV");
    usb_cfg.capture_width = this->declare_parameter<int>("usb.capture_width", 3840);
    usb_cfg.capture_height = this->declare_parameter<int>("usb.capture_height", 1080);
    usb_cfg.fps = this->declare_parameter<double>("usb.fps", 30.0);
    usb_cfg.buffer_size = this->declare_parameter<int>("usb.buffer_size", 1);
    usb_cfg.output_width = this->declare_parameter<int>("usb.output_width", 1280);
    usb_cfg.output_height = this->declare_parameter<int>("usb.output_height", 720);
    usb_cfg.calibration_json = this->declare_parameter<std::string>(
        "usb.calibration_json", "src/wheel_perception/camera/stereo_calibration.json");
    usb_cfg.swap_left_right = this->declare_parameter<bool>("usb.swap_left_right", true);
    usb_cfg.require_calibration_size =
        this->declare_parameter<bool>("usb.require_calibration_size", true);
    usb_cfg.rectify_alpha = this->declare_parameter<double>("usb.rectify_alpha", 0.0);
    usb_cfg.auto_reconnect = this->declare_parameter<bool>("usb.auto_reconnect", true);
    usb_cfg.reconnect_interval_ms =
        this->declare_parameter<int>("usb.reconnect_interval_ms", 500);
    usb_cfg.reconnect_warmup_frames =
        this->declare_parameter<int>("usb.reconnect_warmup_frames", 3);
    usb_cfg.brightness = this->declare_parameter<int>("usb.controls.brightness", -1);
    usb_cfg.contrast = this->declare_parameter<int>("usb.controls.contrast", -1);
    usb_cfg.saturation = this->declare_parameter<int>("usb.controls.saturation", -1);
    usb_cfg.hue = this->declare_parameter<int>("usb.controls.hue", -1);
    usb_cfg.white_balance_automatic =
        this->declare_parameter<int>("usb.controls.white_balance_automatic", -1);
    usb_cfg.white_balance_temperature =
        this->declare_parameter<int>("usb.controls.white_balance_temperature", -1);
    usb_cfg.gamma = this->declare_parameter<int>("usb.controls.gamma", -1);
    usb_cfg.gain = this->declare_parameter<int>("usb.controls.gain", -1);
    usb_cfg.power_line_frequency =
        this->declare_parameter<int>("usb.controls.power_line_frequency", -1);
    usb_cfg.sharpness = this->declare_parameter<int>("usb.controls.sharpness", -1);
    usb_cfg.backlight_compensation =
        this->declare_parameter<int>("usb.controls.backlight_compensation", -1);

    core::EssDepthEngine::Config ess_cfg;
    ess_cfg.engine_path = this->declare_parameter<std::string>(
        "ess.engine_path", "src/wheel_perception/ess/light_ess.engine");
    ess_cfg.plugin_path = this->declare_parameter<std::string>(
        "ess.plugin_path", "src/wheel_perception/ess/plugins/aarch64/ess_plugins.so");
    ess_cfg.input_width = this->declare_parameter<int>("ess.input_width", 480);
    ess_cfg.input_height = this->declare_parameter<int>("ess.input_height", 288);
    ess_cfg.confidence_threshold =
        this->declare_parameter<double>("ess.confidence_threshold", 0.5);
    ess_cfg.min_depth_m = this->declare_parameter<double>("ess.min_depth_m", zed_cfg.depth_min);
    ess_cfg.max_depth_m = this->declare_parameter<double>("ess.max_depth_m", zed_cfg.depth_max);

    // ================= 2.5 远程 ESS 深度参数 (双机模式, zed.depth_source=ess_remote) =================
    // 本机不跑 ESS 模型：把双目拼接图 JPEG 压缩发到目标机，收目标机回传的深度图。
    remote_stereo_topic_ = this->declare_parameter<std::string>(
        "ess_remote.topics.stereo_input", "ess_remote/stereo_input");
    remote_depth_topic_ = this->declare_parameter<std::string>(
        "ess_remote.topics.depth", "ess_remote/depth");
    remote_calib_topic_ = this->declare_parameter<std::string>(
        "ess_remote.topics.calibration", "ess_remote/calibration");
    remote_publish_interval_ms_ = this->declare_parameter<int>(
        "ess_remote.publish_interval_ms", 33); // 约 30Hz
    remote_jpeg_quality_ = this->declare_parameter<int>(
        "ess_remote.jpeg_quality", 85);
    remote_stale_timeout_ms_ = this->declare_parameter<int64_t>(
        "ess_remote.stale_timeout_ms", 500); // 深度超过该时长视为过期
    // true：深度过期也不丢，沿用上一次收到的深度（慢链路低频更新时仍能融合）
    reuse_stale_depth_ = this->declare_parameter<bool>(
        "ess_remote.reuse_stale_depth", true);

    // ================= 3. 感知算法参数 (PerceptionParams) =================
    // 空间剪裁
    params_.min_x = this->declare_parameter<float>("perception.roi.min_x", 0.3f);
    params_.max_x = this->declare_parameter<float>("perception.roi.max_x", 5.0f);
    params_.min_y = this->declare_parameter<float>("perception.roi.min_y", -3.0f);
    params_.max_y = this->declare_parameter<float>("perception.roi.max_y", 3.0f);
    params_.min_z = this->declare_parameter<float>("perception.roi.min_z", -0.2f);
    params_.max_z = this->declare_parameter<float>("perception.roi.max_z", 2.0f);

    // 性能与采样
    params_.step = this->declare_parameter<int>("perception.sample_step", 2);
    params_.voxel_size = this->declare_parameter<float>("perception.voxel_size", 0.05f); // [新增]

    // 几何避障逻辑 (椭圆参数)
    params_.ellipse_x = this->declare_parameter<float>("perception.obstacle.ellipse_x_weight", 6.0f);
    params_.ellipse_y = this->declare_parameter<float>("perception.obstacle.ellipse_y_weight", 100.0f);
    params_.ellipse_thres = this->declare_parameter<float>("perception.obstacle.ellipse_threshold", 50.0f);

    // ================= 4. 动态右侧目标距离参数 =================
    dynamic_aim_enabled_ = this->declare_parameter<bool>("dynamic_aim.enabled", true);
    dynamic_aim_ratio_ = this->declare_parameter<float>("dynamic_aim.ratio", 0.35f);
    dynamic_aim_min_dist_ = this->declare_parameter<float>("dynamic_aim.min_dist", 0.65f);
    dynamic_aim_max_dist_ = this->declare_parameter<float>("dynamic_aim.max_dist", 1.0f);
    dynamic_aim_fallback_dist_ = this->declare_parameter<float>("dynamic_aim.fallback_dist", 1.0f);
    dynamic_aim_min_width_ = this->declare_parameter<float>("dynamic_aim.min_width", 1.0f);
    dynamic_aim_max_width_ = this->declare_parameter<float>("dynamic_aim.max_width", 15.0f);
    dynamic_aim_max_target_step_ = this->declare_parameter<float>("dynamic_aim.max_target_step", 0.05f);
    dynamic_aim_hold_frames_ = this->declare_parameter<int>("dynamic_aim.hold_frames", 10);

    // ================= 初始化模块 =================
    if (!init_camera(zed_cfg, usb_cfg, ess_cfg)) return CallbackReturn::FAILURE;
    
    // 初始化 AI
    if (!init_ai(engine_path)) return CallbackReturn::FAILURE;
    
    int ai_w=0, ai_h=0;
    ai_engine_->getInputResolution(ai_w, ai_h);
    fusion_ = std::make_unique<core::ObstacleFusion>(ai_w, ai_h);

    pub_perception_ = this->create_publisher<wheel_msgs::msg::PerceptionOutput>("perception/output", 10);
    pub_viz_ = this->create_publisher<sensor_msgs::msg::Image>("debug/viz", 10);
    // 左相机调试图：复用道路 mask 叠加结果，并显示道路几何量。
    pub_viz_mouse_left_ = this->create_publisher<sensor_msgs::msg::Image>("/debug/viz_mouse_left", 10);
    pub_bev_fused_ = this->create_publisher<sensor_msgs::msg::Image>("debug/bev_fused", 10);
    pub_cloud_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("zed/point_cloud", 10);

    // 两个调试点云发布者
    pub_cloud_rect_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("perception/debug/cloud_rect", 10);
    pub_cloud_ellipse_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("perception/debug/cloud_ellipse", 10);

    RCLCPP_INFO(get_logger(), "Perception Configured: Voxel=%.2f, Ellipse=%.1fx^2 + %.1fy^2 < %.1f, DynamicAim=%s",
        params_.voxel_size, params_.ellipse_x, params_.ellipse_y, params_.ellipse_thres,
        dynamic_aim_enabled_ ? "true" : "false");
    
    // ZED里程计发布者
    pub_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
    return CallbackReturn::SUCCESS;
}


    CallbackReturn on_activate(const rclcpp_lifecycle::State & state) override {
        pub_perception_->on_activate();
        pub_viz_->on_activate();
        pub_viz_mouse_left_->on_activate();
        pub_cloud_->on_activate();
        pub_cloud_rect_->on_activate();
        pub_cloud_ellipse_->on_activate();
        pub_odom_->on_activate();
        pub_bev_fused_->on_activate();
        if (pub_remote_stereo_) pub_remote_stereo_->on_activate();   // [ess_remote]
        if (pub_remote_calib_) pub_remote_calib_->on_activate();     // [ess_remote]
        timer_ = this->create_wall_timer(std::chrono::milliseconds(15), std::bind(&FusionNode::update_loop, this));
        return LifecycleNode::on_activate(state);
    }

    CallbackReturn on_deactivate(const rclcpp_lifecycle::State & state) override {
        if (timer_) timer_->cancel();
        pub_perception_->on_deactivate();
        pub_viz_->on_deactivate();
        pub_viz_mouse_left_->on_deactivate();
        pub_cloud_->on_deactivate();
        pub_cloud_rect_->on_deactivate();
        pub_cloud_ellipse_->on_deactivate();
        pub_odom_->on_deactivate();
        pub_bev_fused_->on_deactivate();
        if (pub_remote_stereo_) pub_remote_stereo_->on_deactivate(); // [ess_remote]
        if (pub_remote_calib_) pub_remote_calib_->on_deactivate();   // [ess_remote]
        return LifecycleNode::on_deactivate(state);
    }

    CallbackReturn on_cleanup(const rclcpp_lifecycle::State & state) override {
        driver_.reset(); usb_driver_.reset(); ess_engine_.reset(); ai_engine_.reset(); fusion_.reset();
        pub_perception_.reset();
        pub_viz_.reset();
        pub_viz_mouse_left_.reset();
        pub_cloud_.reset();
        pub_cloud_rect_.reset();
        pub_cloud_ellipse_.reset();
        pub_odom_.reset();
        pub_bev_fused_.reset();
        timer_.reset();
        // [ess_remote] 远程链路复位（下一次 configure 重建）
        pub_remote_stereo_.reset();
        pub_remote_calib_.reset();
        sub_remote_depth_.reset();
        latest_remote_depth_.reset();
        remote_calib_published_ = false;
        last_remote_stereo_pub_ns_ = 0;
        
        // [新增] 释放手动开辟的显存
        if (d_rgb_buffer_) {
            cudaFree(d_rgb_buffer_);
            d_rgb_buffer_ = nullptr;
            d_rgb_buffer_size_ = 0;
        }
        if (d_cloud_buffer_) {
            cudaFree(d_cloud_buffer_);
            d_cloud_buffer_ = nullptr;
            d_cloud_buffer_size_ = 0;
        }
        return LifecycleNode::on_cleanup(state);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_rgb_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_depth_; // [修改] 改为订阅深度图
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;          // [新增] 订阅IMU
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;       // [新增] 订阅里程计

    // 缓存最新帧
    sensor_msgs::msg::Image::SharedPtr latest_rgb_;
    sensor_msgs::msg::Image::SharedPtr latest_depth_;   // [修改] 
    sensor_msgs::msg::Imu::SharedPtr latest_imu_;             // [新增]
    nav_msgs::msg::Odometry::SharedPtr latest_odom_;          // [新增]
    std::mutex frame_mutex_;
    
    // [新增] 用于保存 CPU->GPU 拷贝的显存空间
    void* d_rgb_buffer_ = nullptr;
    size_t d_rgb_buffer_size_ = 0;
    // [新增] 用于保存点云的 CPU->GPU 拷贝显存空间
    void* d_cloud_buffer_ = nullptr;
    size_t d_cloud_buffer_size_ = 0;

// 简单的指数移动平均 (EMA) 状态变量
    float ema_dist_ = 0.0f;
    float ema_angle_ = 0.0f;
    bool first_frame_ = true;
    float last_valid_road_width_ = 0.0f;
    float last_target_right_distance_ = 1.0f;
    int road_width_invalid_frames_ = 0;
    
    // EMA 系数 (0.0~1.0)，越小越平滑，但也越迟钝。0.3 是个不错的折中。
    const float EMA_ALPHA = 0.3f;

    void publish_odometry(const core::ZedGpuFrame& frame) {
        if (!pub_odom_ || !pub_odom_->is_activated() || !flag_odom) {
            // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,"ffff%d",(int)flag_odom);
            return;
        }

        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = rclcpp::Time(static_cast<int64_t>(frame.timestamp_ns));
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";

        odom_msg.pose.pose.position.x = frame.pose_x;
        odom_msg.pose.pose.position.y = frame.pose_y;
        odom_msg.pose.pose.position.z = frame.pose_z;
        odom_msg.pose.pose.orientation.x = frame.quat_x;
        odom_msg.pose.pose.orientation.y = frame.quat_y;
        odom_msg.pose.pose.orientation.z = frame.quat_z;
        odom_msg.pose.pose.orientation.w = frame.quat_w;

        pub_odom_->publish(std::move(odom_msg));
    }

    void rgb_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_rgb_ = msg;
    }

    // [修改] 对应 Depth 话题的回调函数
    void depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_depth_ = msg;
    }
    
    // [新增] IMU 回调函数
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_imu_ = msg;
        current_yaw_ = msg.get()->orientation.z; // 简单假设航向角直接来自 IMU 的 z 轴旋转分量，实际可能需要转换
        // RCLCPP_INFO(get_logger(), "yaw = %f", current_yaw_);
    }

    // [新增] Odometry 回调函数
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_odom_ = msg;
        // RCLCPP_INFO(get_logger(), "Received Odometry: Pos(%.2f, %.2f, %.2f) Orient(%.2f, %.2f, %.2f, %.2f)", 
        //     msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z,
        //     msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
        current_x_ = msg->pose.pose.position.x;
        current_y_ = msg->pose.pose.position.y;
        // RCLCPP_INFO(get_logger(), "Received Odometry: Pos(%.2f, %.2f)", current_x_, current_y_);
    }
    
void update_loop() {
        if (!rclcpp::ok()) return;
        if (!driver_ && !usb_driver_ && !use_dataset_mode_) return;

        // [计时点 0] 帧开始
        auto t0 = std::chrono::high_resolution_clock::now();
        core::ZedGpuFrame frame;

        if (!use_dataset_mode_) {
            if (camera_type_ == "usb") {
                if (!usb_driver_ || !usb_driver_->grab(frame)) {
                    return;
                }
                if (!run_ess_depth(frame)) {
                    return;
                }
            } else {
                if (!driver_ || !driver_->grab(frame)) {
                    return;
                }
                publish_odometry(frame);
            }
        }

        void* ai_input_dev_ptr = nullptr; // [新增] 指向最终送到推理模型的 GPU 显存
        void* cloud_input_dev_ptr = nullptr;
        int img_w = 0, img_h = 0;
        bool has_depth_input = false;

        // [BLOCK 1] 取图 (Grab)
        if (use_dataset_mode_) {
            // 注意：这是阻塞操作，会等待相机帧率 (比如 15ms or 33ms)
            // 如果这里耗时高，说明是相机帧率限制，不是代码慢
            sensor_msgs::msg::Image::SharedPtr rgb_msg;
            sensor_msgs::msg::Image::SharedPtr depth_msg;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                if (!latest_rgb_) return; // RGB 未就绪时无法进行模型推理
                rgb_msg   = latest_rgb_;
                depth_msg = latest_depth_;
            }
            // 1. 将 ROS 话题消息转成 CPU cv::Mat
            auto cv_img = cv_bridge::toCvShare(rgb_msg, "bgra8"); // python 节点发的是 bgra8, 4通道
            img_w = cv_img->image.cols;
            img_h = cv_img->image.rows;

            // ================== [新增: 为 AI 进行 Resize] ==================
            int ai_w = 0, ai_h = 0;
            ai_engine_->getInputResolution(ai_w, ai_h);

            // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "DATE Inference Input: GPU Ptr=%p, Src Res=%dx%d, Infer Res=%dx%d", 
            // ai_input_dev_ptr, img_w, img_h, ai_w, ai_h);
            
            cv::Mat ai_img_resized;
            // 调整到 640x480 或引擎需要的任何尺寸
            if (ai_w > 0 && ai_h > 0) {
                cv::resize(cv_img->image, ai_img_resized, cv::Size(ai_w, ai_h), 0, 0, cv::INTER_LINEAR);
            } else {
                ai_img_resized = cv_img->image;
            }

            size_t rgb_bytes = ai_img_resized.cols * ai_img_resized.rows * 4; 
            if (d_rgb_buffer_size_ < rgb_bytes) {
                if (d_rgb_buffer_) cudaFree(d_rgb_buffer_);
                cudaMalloc(&d_rgb_buffer_, rgb_bytes);
                d_rgb_buffer_size_ = rgb_bytes;
            }
            cudaMemcpy(d_rgb_buffer_, ai_img_resized.data, rgb_bytes, cudaMemcpyHostToDevice);
            ai_input_dev_ptr = d_rgb_buffer_;
            // ===============================================================

            // 2. 处理深度图转点云 (可选)。RGB-only dataset 也允许推理和 mask 可视化。
            if (depth_msg) {
                cv_bridge::CvImageConstPtr cv_depth;
                try {
                    if (depth_msg->encoding == "16UC1" || depth_msg->encoding == "32FC1") {
                        cv_depth = cv_bridge::toCvShare(depth_msg, depth_msg->encoding);
                    } else {
                        cv_depth = cv_bridge::toCvShare(depth_msg, "32FC1"); 
                    }
                } catch (cv_bridge::Exception& e) {
                    RCLCPP_ERROR(get_logger(), "depth cv_bridge error: %s", e.what());
                    return;
                }

                if (cv_depth->image.cols != img_w || cv_depth->image.rows != img_h) {
                    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                        "Depth size (%dx%d) does not match RGB size (%dx%d); skip depth-dependent fusion.",
                        cv_depth->image.cols, cv_depth->image.rows, img_w, img_h);
                } else {
                    size_t cloud_bytes = img_w * img_h * sizeof(float4);
                    if (d_cloud_buffer_size_ < cloud_bytes) {
                        if (d_cloud_buffer_) cudaFree(d_cloud_buffer_);
                        cudaMalloc(&d_cloud_buffer_, cloud_bytes);
                        d_cloud_buffer_size_ = cloud_bytes;
                    }

                    std::vector<float4> host_cloud(img_w * img_h);
                    // 默认参数 (如果有 driver 则是实车相机参数)
                    float fx = 527.3f, fy = 527.3f, cx = img_w / 2.0f, cy = img_h / 2.0f;
            if (driver_) {
                auto params = driver_->getCameraParams();
                if (params.fx > 10.0f) { fx = params.fx; fy = params.fy; cx = params.cx; cy = params.cy; }
            } else if (usb_driver_) {
                auto params = usb_driver_->getCameraParams();
                if (params.fx > 10.0f) { fx = params.fx; fy = params.fy; cx = params.cx; cy = params.cy; }
            }

                    bool is_16u = (cv_depth->image.type() == CV_16UC1);
                    float* depth_f = is_16u ? nullptr : (float*)cv_depth->image.data;
                    uint16_t* depth_u = is_16u ? (uint16_t*)cv_depth->image.data : nullptr;

                    for (int v = 0; v < img_h; ++v) {
                        for (int u = 0; u < img_w; ++u) {
                            float d = is_16u ? (depth_u[v * img_w + u] * 0.001f) : depth_f[v * img_w + u];
                            float4 pt;
                            // 在 Z_UP_X_FWD 坐标系下：X向前, Y向左, Z向上
                            if (d > 0.1f && d < 20.0f && std::isfinite(d)) {
                                pt.x = d;
                                pt.y = d * (cx - u) / fx;
                                pt.z = d * (cy - v) / fy;
                                pt.w = 0.0f; 
                            } else {
                                pt.x = 0; pt.y = 0; pt.z = 0; pt.w = 0;
                            }
                            host_cloud[v * img_w + u] = pt;
                        }
                    }
                    // 拷贝至 GPU 显存点云
                    cudaMemcpy(d_cloud_buffer_, host_cloud.data(), cloud_bytes, cudaMemcpyHostToDevice);
                    cloud_input_dev_ptr = d_cloud_buffer_;
                    has_depth_input = true;
                }
            }
        }
        else {
            img_w = frame.width;
            img_h = frame.height;
            cloud_input_dev_ptr = frame.cloud_ptr_dev;
            has_depth_input = (cloud_input_dev_ptr != nullptr);

            int ai_w = 0, ai_h = 0;
            ai_engine_->getInputResolution(ai_w, ai_h);

            // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "RELITY Inference Input: GPU Ptr=%p, Src Res=%dx%d, Infer Res=%dx%d", 
            // ai_input_dev_ptr, img_w, img_h, ai_w, ai_h);

            // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "RESIZE NEEDED for AI Inference");
            // 1. 把 ZED 显存里的图像拉回 CPU
            cv::Mat zed_img(img_h, img_w, CV_8UC4);
            cudaMemcpy2D(
                zed_img.data, static_cast<size_t>(img_w * 4),
                frame.rgb_ptr_dev, static_cast<size_t>(frame.rgb_step_bytes),
                static_cast<size_t>(img_w * 4), static_cast<size_t>(img_h),
                cudaMemcpyDeviceToHost);

            // 2. OpenCV Resize
            cv::Mat ai_img_resized;
            if (ai_w > 0 && ai_h > 0 && (ai_w != img_w || ai_h != img_h)) {
                cv::resize(zed_img, ai_img_resized, cv::Size(ai_w, ai_h), 0, 0, cv::INTER_LINEAR);
            } else {
                ai_img_resized = zed_img;
            }

            // 3. 把 Resize 后的图像拷回自己申请的显存，准备喂给模型
            size_t rgb_bytes = ai_img_resized.cols * ai_img_resized.rows * 4; 
            if (d_rgb_buffer_size_ < rgb_bytes) {
                if (d_rgb_buffer_) cudaFree(d_rgb_buffer_);
                cudaMalloc(&d_rgb_buffer_, rgb_bytes);
                d_rgb_buffer_size_ = rgb_bytes;
            }
            cudaMemcpy(d_rgb_buffer_, ai_img_resized.data, rgb_bytes, cudaMemcpyHostToDevice);

            
            
            ai_input_dev_ptr = d_rgb_buffer_;
        }
        
        
        // RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 100,"fff");
        auto t1 = std::chrono::high_resolution_clock::now();

        // [BLOCK 2] AI 推理
        int ai_infer_w = 0, ai_infer_h = 0;
        ai_engine_->getInputResolution(ai_infer_w, ai_infer_h);
        
        // 现在无论哪种模式，ai_input_dev_ptr 都已经是正确的尺寸了
        ai_engine_->infer(ai_input_dev_ptr, ai_infer_w, ai_infer_h); 
        
        void* ai_output = ai_engine_->getOutputTensor("output"); // 获取显存指针
        fusion_->processSegmentation(ai_output);         // 执行 ArgMax
        auto t2 = std::chrono::high_resolution_clock::now();

        int valid_count = 0;
        SceneMetrics metrics;
        std::vector<core::RoadEdgePoint> edges;
        std::vector<core::RoadEdgePoint> left_edges;
        std::vector<float4> filtered_pts;

        if (has_depth_input) {
            // [BLOCK 3] GPU 滤波 (CUDA Kernel)
            // 这里应该极快 (< 2ms)
            valid_count = fusion_->filterCloud(cloud_input_dev_ptr, img_w, img_h, params_);
            
            // [BLOCK 4] 提取车道线 (CUDA Kernel)
            fusion_->extractRoadEdges(cloud_input_dev_ptr, img_w, img_h, params_, edges, left_edges);

            fusion_->retrieveFilteredCloud(filtered_pts); // 获取带有 .w 分类标记的混合点云
        } else {
            metrics.min_dist = 10.0f;
            metrics.min_point.x = 10.0f;
            metrics.road_dist = 10.0f;
        }

        auto t3 = std::chrono::high_resolution_clock::now();

        // RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200,"minx=%.2f miny=%.2f",x_min, y_min);
        // RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200,"x=%.2f y=%.2f yaw=%.2f",current_x_, current_y_, current_yaw_);

        // [BLOCK 5] 点云发布 (CPU 搬运) --- 重点嫌疑对象!!!
        // 这里涉及 D2H 拷贝 + for 循环填充 ROS 消息
        /////////////////////////////////////-------------------------------------------------------------------------
        // if(!flag_obstacle) process_blind_zone_and_bev(cloud_input_dev_ptr, img_w, img_h);
        ///////////////////////////////////////-------------------------------------------------------------------------

        // 清空临时容器
        cloud_rect_vec_.clear();
        cloud_ellipse_vec_.clear();

        // 定义距离变量
        float min_dist_rect = 100.0f;    // 给 LQR 用
        geometry_msgs::msg::Point min_pt_rect;
        bool has_rect = false;

        float min_dist_ellipse = 2.0f; // 给 避障 用
        geometry_msgs::msg::Point min_pt_ellipse;
        bool has_ellipse = false;

        // 遍历拆分 (CPU 操作，非常快)
        for (const auto& pt : filtered_pts) {
            int category = (int)pt.w; // 读取 CUDA 写入的分类标记

            // --- 列表 1: 矩形/路肩 (标记位 1) ---
            if (category & 1) {
                cloud_rect_vec_.push_back(pt);
                // 计算右侧距离 (加权 x^2 + 4y^2)
                float d_sq = pt.x * pt.x + 4.0f * pt.y * pt.y;
                if (d_sq < min_dist_rect) {
                    min_dist_rect = d_sq;
                    min_pt_rect.x = pt.x; min_pt_rect.y = pt.y; min_pt_rect.z = pt.z;
                    has_rect = true;
                }
            }

            // --- 列表 2: 椭圆/障碍 (标记位 2) ---
            if (category & 2) {
                cloud_ellipse_vec_.push_back(pt);
                // 计算避障距离
                float d_sq = pt.x;
                if (d_sq < min_dist_ellipse) {
                    min_dist_ellipse = d_sq;
                    min_pt_ellipse.x = pt.x; min_pt_ellipse.y = pt.y; min_pt_ellipse.z = pt.z;
                    has_ellipse = true;
                }
            }
        }

        // [BLOCK 6] 发布调试点云 (可选，用于 Rviz)
        // 辅助 Lambda: 发布 float4 向量为 PointCloud2
        auto publish_cloud_helper = [&](auto& publisher, const std::vector<float4>& pts, const std::string& frame_id) {
            if (publisher->get_subscription_count() == 0 || pts.empty()) return;
            
            auto msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
            msg->header.stamp = this->get_clock()->now();
            msg->header.frame_id = frame_id;
            msg->height = 1; msg->width = pts.size(); msg->is_dense = false;
            sensor_msgs::PointCloud2Modifier modifier(*msg);
            modifier.setPointCloud2FieldsByString(1, "xyz");
            modifier.resize(msg->width);
            sensor_msgs::PointCloud2Iterator<float> iter_x(*msg, "x"), iter_y(*msg, "y"), iter_z(*msg, "z");
            
            for (const auto& p : pts) {
                *iter_x = p.x; *iter_y = p.y; *iter_z = p.z;
                ++iter_x; ++iter_y; ++iter_z;
            }
            publisher->publish(std::move(msg));
        };

        
        publish_cloud_helper(pub_cloud_, filtered_pts, "zed_left_camera_frame");
        publish_cloud_helper(pub_cloud_rect_, cloud_rect_vec_, "zed_left_camera_frame");
        publish_cloud_helper(pub_cloud_ellipse_, cloud_ellipse_vec_, "zed_left_camera_frame");
        auto t4 = std::chrono::high_resolution_clock::now();

        // [BLOCK 7] 填充 Metrics (替代 analyze_scene 的部分功能)

        // 1. 填充 LQR 数据 (来自矩形列表)
        if (has_rect) {
            metrics.road_dist = std::sqrt(min_dist_rect); // 开根号还原为米
            // 这里如果还需要 road_yaw_error，需要依赖 [BLOCK 4] 提取的车道线结果
            // 下面的 analyze_scene 会处理 yaw
        } else {
            metrics.road_dist = 10.0f;
        }

        // 2. 填充避障数据 (来自椭圆列表)
        if (has_ellipse) {
            metrics.min_dist = min_dist_ellipse;
            metrics.min_point = min_pt_ellipse;
        } else {
            metrics.min_dist = 10.0f;
            metrics.min_point.x = 10.0f; // 设远一点
        }
        std::vector<core::Obstacle3DStat> obstacles; // 保持为空即可，或者如果你想可视化方框，可以在这里做聚类
        analyze_scene(obstacles, edges, left_edges, metrics);

        if(!metrics.has_line) out_left0 = true;
        else out_left0 = false;

        if(metrics.is_vertical) out_right0 = true;
        else out_right0 = false;
        publish_perception_msg(obstacles, metrics);
        
        static int cnt = 0;
        bool is_viz_published = false;
        if (++cnt % 1 == 0) { // 降频显示
            publish_visualization(obstacles, edges, metrics);
            is_viz_published = true;
        }
        auto t5 = std::chrono::high_resolution_clock::now();

        // =========================================================
        // [性能报告] 每 60 帧 (约1秒) 打印一次
        // =========================================================
        static int report_cnt = 0;
        if (++report_cnt % 60 == 0) {
            double ms_grab   = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double ms_ai     = std::chrono::duration<double, std::milli>(t2 - t1).count();
            double ms_gpu    = std::chrono::duration<double, std::milli>(t3 - t2).count();
            double ms_cloud  = std::chrono::duration<double, std::milli>(t4 - t3).count(); // 重点看这个
            double ms_logic  = std::chrono::duration<double, std::milli>(t5 - t4).count();
            double ms_total  = std::chrono::duration<double, std::milli>(t5 - t0).count();
            double fps       = 1000.0 / ms_total;

            RCLCPP_INFO(get_logger(), 
                "\n=== [Performance Profiling] ===\n"
                "1. ZED Grab      : %5.2f ms (Camera IO)\n"
                "2. AI Infer      : %5.2f ms (TensorRT)\n"
                "3. Fusion Kernel : %5.2f ms (CUDA Filter + Edge)\n"
                "4. Cloud Pub     : %5.2f ms (Copy + Loop) [Count: %d] %s\n"
                "5. Vis & Logic   : %5.2f ms (OpenCV) %s\n"
                "--------------------------------\n"
                "   TOTAL TIME    : %5.2f ms\n"
                "   CURRENT FPS   : %5.1f Hz\n"
                "================================",
                ms_grab, ms_ai, ms_gpu, 
                ms_cloud, valid_count, (is_viz_published ? "[ACTIVE]" : "[SKIP]"),
                ms_logic, (is_viz_published ? "[ACTIVE]" : "[SKIP]"),
                ms_total, fps);
        }
    }

    // 辅助函数: 3D -> 2D 投影
    cv::Point project_pt(float x, float y, float z, const core::ZedDriver::CameraParams& p) {
        if (x < 0.1f) return {-1000, -1000};
        int u = p.fx * (-y / x) + p.cx;
        int v = p.fy * (-z / x) + p.cy;
        return {u, v}; // 这里不做 clamp，交给调用的地方或者画图函数处理
    }
    float median_or_zero(std::vector<float>& values) {
        if (values.empty()) return 0.0f;
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    void update_dynamic_target_from_edges(const std::vector<core::RoadEdgePoint>& right_edges,
                                          const std::vector<core::RoadEdgePoint>& left_edges,
                                          SceneMetrics& metrics)
    {
        metrics.target_right_distance = last_target_right_distance_;
        if (!dynamic_aim_enabled_) return;

        std::vector<float> widths;
        std::vector<float> left_dists;
        const size_t n = std::min(right_edges.size(), left_edges.size());
        for (size_t i = 0; i < n; ++i) {
            const auto& right = right_edges[i];
            const auto& left = left_edges[i];
            if (!right.valid || !left.valid) continue;
            // 道路宽度使用较远处的可见边界；LQR 右边界拟合范围保持不变。
            if (right.x < 1.0f || right.x > 15.0f || left.x < 1.0f || left.x > 15.0f) continue;
            if (std::abs(right.x - left.x) > 1.2f) continue;
            if (right.y >= -0.05f || left.y <= 0.05f) continue;

            const float width = left.y - right.y;
            if (width < dynamic_aim_min_width_ || width > dynamic_aim_max_width_) continue;
            widths.push_back(width);
            left_dists.push_back(left.y);
        }

        float measured_width = median_or_zero(widths);
        const bool enough_samples = widths.size() >= 1;
        bool width_trusted = enough_samples && measured_width > 0.0f;

        // 左侧动态障碍物或最大连通域收缩时，宽度会突然变小；这种帧先冻结目标距离。
        if (width_trusted && last_valid_road_width_ > 0.0f &&
            measured_width < last_valid_road_width_ - 0.6f) {
            width_trusted = false;
        }

        if (width_trusted) {
            road_width_invalid_frames_ = 0;
            last_valid_road_width_ = measured_width;

            float raw_target = std::clamp(
                measured_width * dynamic_aim_ratio_,
                dynamic_aim_min_dist_,
                dynamic_aim_max_dist_);
            const float delta = raw_target - last_target_right_distance_;
            const float max_step = std::max(0.0f, dynamic_aim_max_target_step_);
            if (std::abs(delta) > max_step) {
                raw_target = last_target_right_distance_ + std::copysign(max_step, delta);
            }

            last_target_right_distance_ = raw_target;
            metrics.has_road_width = true;
            metrics.road_width = measured_width;
            metrics.left_distance = median_or_zero(left_dists);
            metrics.target_right_distance = last_target_right_distance_;
        } else {
            road_width_invalid_frames_++;
            if (road_width_invalid_frames_ > dynamic_aim_hold_frames_) {
                last_target_right_distance_ = dynamic_aim_fallback_dist_;
            }

            metrics.has_road_width = false;
            metrics.road_width = last_valid_road_width_;
            metrics.target_right_distance = last_target_right_distance_;
        }

        RCLCPP_INFO_THROTTLE(
            get_logger(), *this->get_clock(), 500,
            "DynamicAim: valid=%s width=%.2f target=%.2f left=%.2f samples=%zu",
            metrics.has_road_width ? "true" : "false",
            metrics.road_width,
            metrics.target_right_distance,
            metrics.left_distance,
            widths.size());
    }

// [重写] 核心分析逻辑：复刻 Python 的双区间均值法
    void analyze_scene(const std::vector<core::Obstacle3DStat>& obstacles,
	                   const std::vector<core::RoadEdgePoint>& edges,
                       const std::vector<core::RoadEdgePoint>& left_edges,
	                   SceneMetrics& metrics)
    {
        (void)obstacles;
        update_dynamic_target_from_edges(edges, left_edges, metrics);

        // --- B. 道路边缘拟合 (关键算法更新) ---
        std::vector<cv::Point3f> valid_pts;
        
        // 1. 收集有效点
        // 注意：ZED SDK 输出的 PointCloud 已经是 X-Forward, Y-Left
        // 所以右侧路沿应该是 Y < 0
        for (const auto& p : edges) {
            // 过滤规则：
            // x > 0.5: 去掉车头盲区
            // x < 15.0: 太远的不准
            // y < 0: 只看右边
            if (p.valid && p.x > 0.5f && p.x < 15.0f && p.y < 0.0f) {
                valid_pts.push_back(cv::Point3f(p.x, p.y, p.z));
            }
        }

        // 2. 双区间算法 (Double Interval)
        // 由于 edges 是按图像行扫描的 (Top -> Bottom)，对应的物理空间大约是 (Far -> Near)
        // 所以 valid_pts[0] 是最远的，valid_pts.back() 是最近的。
        size_t n = valid_pts.size();
        
        if (n > 10) {
            // Python逻辑: 远端区间 20%~40% (Far), 近端区间 60%~80% (Near)
            size_t i1_start = n * 0.20;
            size_t i1_end   = n * 0.40;
            
            size_t i2_start = n * 0.60;
            size_t i2_end   = n * 0.80;

            // 简单的鲁棒性检查
            if (i1_end < n && i2_end < n && i1_end > i1_start && i2_end > i2_start) {
                
                // 计算重心 (Centroids)
                auto get_mean = [&](size_t start, size_t end) {
                    float sum_x = 0, sum_y = 0, sum_z = 0;
                    for (size_t i = start; i < end; ++i) {
                        sum_x += valid_pts[i].x;
                        sum_y += valid_pts[i].y;
                        sum_z += valid_pts[i].z;
                    }
                    float count = (float)(end - start);
                    return cv::Point3f(sum_x/count, sum_y/count, sum_z/count);
                };

                cv::Point3f p_far = get_mean(i1_start, i1_end);  // 远端重心
                cv::Point3f p_near = get_mean(i2_start, i2_end); // 近端重心

                // 3. 计算向量与角度
                // 向量方向：从 近 -> 远
                float vx = p_far.x - p_near.x;
                float vy = p_far.y - p_near.y;

                float raw_yaw = std::atan2(vy, vx); 

                // ====== [新增判断逻辑] ======
                // 如果是相对于车头坐标系（X前Y左）：
                // atan2 返回的范围是 [-π, π]。
                // 如果这条线相对于 X 轴接近 90 度 (π/2) 或 -90 度 (-π/2)：
                // 我们可以认为在 2D 平面上它是“垂直”的 (比如横着的一堵墙或路沿急转弯)
                // 假设我们容忍的误差范围是 ±10 度 (10 * π / 180 ≈ 0.174 弧度)
                
                // ============================

                // 4. 计算横向距离 (Distance to Line)
                // 直线方程： (vy)*X - (vx)*Y + C = 0
                // 代入点 p_near 计算 C: C = vx*y0 - vy*x0
                // 原点(0,0)到直线的距离: |C| / sqrt(vx^2 + vy^2)
                // 这里的 distance 是原点到直线的垂线距离
                float norm = std::hypot(vx, vy);
                float raw_dist = 0.0f;
                if (norm > 1e-4) {
                    raw_dist = std::abs(vx * p_near.y - vy * p_near.x) / norm;
                }
                // RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200, 
                // "Road edge detected. Distance: %f, Yaw: %f", raw_dist, raw_yaw);
                // 5. EMA 平滑 (防止控制抖动)
                if (first_frame_) {
                    ema_dist_ = raw_dist;
                    ema_angle_ = raw_yaw;
                    first_frame_ = false;
                } else {
                    ema_dist_ = EMA_ALPHA * raw_dist + (1.0f - EMA_ALPHA) * ema_dist_;
                    ema_angle_ = EMA_ALPHA * raw_yaw + (1.0f - EMA_ALPHA) * ema_angle_;
                }

                // 6. 填充结果
                metrics.has_line = true;
                metrics.road_dist = ema_dist_;
                metrics.road_yaw_rad = ema_angle_;

                if (metrics.road_dist <= 0.05f) {
                    metrics.is_vertical = true;
                } else {
                    metrics.is_vertical = false;
                }
                if(abs(valid_pts[n-1].y) > raw_dist + 1.0f) metrics.is_vertical = true;
                // 记录调试点
                metrics.debug_pt1.x = p_far.x; metrics.debug_pt1.y = p_far.y; metrics.debug_pt1.z = p_far.z;
                metrics.debug_pt2.x = p_near.x; metrics.debug_pt2.y = p_near.y; metrics.debug_pt2.z = p_near.z;
                
                return; // 成功返回
            }
        }
        else metrics.has_line = false;
        // 如果点不够或者拟合失败
        // 保持上一帧的值或者归零，这里选择安全归零
        // metrics.road_dist = 0; 
    }

    //盲区填补函数调用
    void process_blind_zone_and_bev(void* cloud_input_dev_ptr, int img_w, int img_h) 
    {
        bool calc_left = false, calc_right = false;
        float calc_right_dist = 0.0f;
        
        // 新增用于接收 BEV 的容器
        std::vector<uint8_t> fused_bev_data;

        // 将点云指针、尺寸、当前位姿传入底层 GPU
        fusion_->processBlindZoneOnGPU(
            cloud_input_dev_ptr, img_w, img_h,
            current_x_, current_y_, current_yaw_,
            calc_left, calc_right, calc_right_dist,
            fused_bev_data // 传入新参数
        );

        out_left0 = calc_left;
        out_right0 = calc_right;
        right_right = calc_right_dist;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 200, 
            "Blind Zone Check: Left=%s, Right=%s, RightDist=%.2f", 
            calc_left ? "YES" : "no", calc_right ? "YES" : "no", calc_right_dist);
        
        if (pub_bev_fused_ && pub_bev_fused_->is_activated() && !fused_bev_data.empty()) {
            cv::Mat gray(80, 80, CV_8UC1, fused_bev_data.data());
            cv::Mat color(80, 80, CV_8UC3, cv::Scalar(0, 0, 0)); 

            cv::Mat road_mask = (gray == 0);
            cv::Mat bg_mask = (gray == 255);
            cv::Mat obs_mask; 
            cv::bitwise_not(road_mask | bg_mask, obs_mask); 

            color.setTo(cv::Scalar(128, 0, 0), road_mask);     
            color.setTo(cv::Scalar(255, 200, 100), obs_mask);  

            cv::Mat vis;
            cv::resize(color, vis, cv::Size(80 * 5, 80 * 5), 0, 0, cv::INTER_NEAREST);
            cv::flip(vis, vis, -1); 
            
            auto img_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", vis).toImageMsg();
            img_msg->header.stamp = this->get_clock()->now();
            pub_bev_fused_->publish(*img_msg);
        }
    }

    void publish_perception_msg(const std::vector<core::Obstacle3DStat>& obstacles, const SceneMetrics& metrics) {
        if (!pub_perception_) return;
        
        auto msg = std::make_unique<wheel_msgs::msg::PerceptionOutput>();
        msg->header.stamp = this->get_clock()->now();
        msg->header.frame_id = "base_link"; // 我们已经转换到了车体坐标系

        // 1. 紧急避障信息
        msg->min_distance = metrics.min_dist;
        msg->min_point = metrics.min_point;

        // 2. 巡线与 LQR 信息
        msg->has_road_edge = metrics.has_line;
        msg->right_distance = metrics.road_dist;     // 对应 .msg 中的 right_distance
        msg->road_yaw_error = metrics.road_yaw_rad;  // 对应 .msg 中的 road_yaw_error (弧度)
        msg->has_road_width = metrics.has_road_width;
        msg->left_distance = metrics.left_distance;
        msg->road_width = metrics.road_width;
        msg->target_right_distance = metrics.target_right_distance;

        // 3. 调试信息
        msg->debug_line_pt1 = metrics.debug_pt1;
        msg->debug_line_pt2 = metrics.debug_pt2;

        // 4. 完整的障碍物列表
        for (const auto& obs : obstacles) {
            wheel_msgs::msg::ObstacleInfo info;
            info.class_id = 1; // 暂时默认为 1
            info.confidence = 1.0f;
            
            // 直接传递，已经是 ROS 坐标系
            info.position.x = obs.min_x;
            info.position.y = obs.min_y;
            info.position.z = obs.min_z;
            
            // 粗略估算尺寸
            info.size.x = obs.max_x - obs.min_x;
            info.size.y = obs.max_y - obs.min_y;
            info.size.z = obs.max_z - obs.min_z;

            msg->obstacles.push_back(info);
        }

        pub_perception_->publish(std::move(msg));
    }

    void publish_visualization(const std::vector<core::Obstacle3DStat>& obstacles, 
                               const std::vector<core::RoadEdgePoint>& edges, 
                               const SceneMetrics& metrics) 
    {
        if (!rclcpp::ok()) return;
        // 两个调试话题共用一次渲染，均无人订阅时跳过 CPU mask 下载与绘制。
        if (this->count_subscribers("debug/viz") == 0 &&
            this->count_subscribers("/debug/viz_mouse_left") == 0) return;
        cv::Mat img;
        
        if (use_dataset_mode_) {
            sensor_msgs::msg::Image::SharedPtr rgb_msg;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                rgb_msg = latest_rgb_;
            }
            if (!rgb_msg) return;
            try {
                // 将接收到的图像转换为 bgr8 供 OpenCV 渲染
                img = cv_bridge::toCvCopy(rgb_msg, "bgr8")->image;
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
        } else {
            if (driver_) {
                driver_->retrieveImage(img);
            } else if (usb_driver_) {
                usb_driver_->retrieveImage(img);
            }
        }

        if (img.empty()) return;

        // 1. 下载 Mask 到 CPU
        // 假设您在 ObstacleFusion 里有 downloadMask 接口 (上次代码里有)
        int ai_w = 0, ai_h = 0;
        ai_engine_->getInputResolution(ai_w, ai_h);
        std::vector<unsigned char> mask_cpu(ai_w * ai_h);
        fusion_->downloadMask(mask_cpu.data());

        // 2. 制作“仅道路”mask
        cv::Mat mask_mat(ai_h, ai_w, CV_8UC1, mask_cpu.data());

        // 假设 0 = road
        cv::Mat road_mask = (mask_mat == 0);

        // 3. 尺寸对齐到原图
        cv::Mat road_mask_resized;
        cv::resize(road_mask, road_mask_resized, img.size(), 0, 0, cv::INTER_NEAREST);

        // 4. 只在道路区域叠加绿色 mask
        cv::Mat overlay = img.clone();
        overlay.setTo(cv::Scalar(0, 255, 0), road_mask_resized);  // BGR，绿色

        // 只对道路区域做透明融合
        cv::Mat blended;
        cv::addWeighted(img, 0.7, overlay, 0.3, 0.0, blended);
        blended.copyTo(img, road_mask_resized);

        // 简单画一下路沿
        core::ZedDriver::CameraParams params;
        if (driver_) {
            params = driver_->getCameraParams();
        } else if (usb_driver_) {
            const auto p = usb_driver_->getCameraParams();
            params.fx = p.fx; params.fy = p.fy;
            params.cx = p.cx; params.cy = p.cy;
            params.baseline = p.baseline;
        } else {
            // 给一个默认的假内参避免计算崩溃
            params.fx = 521.9f; params.fy = 521.9f; 
            params.cx = 637.5f; params.cy = 363.4f; 
        }
        if (metrics.has_line) {
             for(const auto& p : edges) {
                 if(p.valid && p.x > 0.5) {
                     cv::Point pt = project_pt(p.x, p.y, p.z, params);
                     if(pt.x>0 && pt.x<img.cols && pt.y>0 && pt.y<img.rows) {
                         cv::circle(img, pt, 2, {0,255,0}, -1);
                     }
                 }
             }
        }

        // 左上角集中显示控制目标与道路测量结果，便于现场对照调试。
        char target_buf[64];
        char right_buf[64];
        char width_buf[64];
        std::snprintf(target_buf, sizeof(target_buf),
                      "Target Dist: %.2f m", metrics.target_right_distance);
        std::snprintf(right_buf, sizeof(right_buf),
                      "Right Dist: %.2f m", metrics.road_dist);
        if (metrics.has_road_width) {
            std::snprintf(width_buf, sizeof(width_buf),
                          "Road Width: %.2f m", metrics.road_width);
        } else {
            std::snprintf(width_buf, sizeof(width_buf), "Road Width: N/A");
        }
        cv::putText(img, target_buf, {20, 40}, cv::FONT_HERSHEY_SIMPLEX,
                    0.9, {0, 255, 255}, 2);
        cv::putText(img, right_buf, {20, 75}, cv::FONT_HERSHEY_SIMPLEX,
                    0.9, {0, 255, 0}, 2);
        cv::putText(img, width_buf, {20, 110}, cv::FONT_HERSHEY_SIMPLEX,
                    0.9, metrics.has_road_width ? cv::Scalar(255, 255, 0)
                                                : cv::Scalar(0, 165, 255),
                    2);
        
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img).toImageMsg();
        pub_viz_->publish(*msg);
        pub_viz_mouse_left_->publish(*msg);
    }

    bool init_camera(
        const core::ZedDriver::Config& zed_cfg,
        const core::UsbStereoDriver::Config& usb_cfg,
        const core::EssDepthEngine::Config& ess_cfg) {
        if(use_dataset_mode_){  //新增
            sub_rgb_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/rgb_image", rclcpp::SensorDataQoS(),
            std::bind(&FusionNode::rgb_callback, this, std::placeholders::_1));
            sub_depth_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/depth_image", rclcpp::SensorDataQoS(),
                std::bind(&FusionNode::depth_callback, this, std::placeholders::_1));
            // [新增]
            sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
                "/zed/imu", rclcpp::SensorDataQoS(),
                std::bind(&FusionNode::imu_callback, this, std::placeholders::_1));
            sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
                "/zed/odom", rclcpp::SensorDataQoS(),
                std::bind(&FusionNode::odom_callback, this, std::placeholders::_1));
            return true;
        }

        if (camera_type_ == "usb") {
            if (depth_source_ != "ess" && depth_source_ != "ess_remote") {
                RCLCPP_ERROR(
                    get_logger(),
                    "USB camera mode requires zed.depth_source=ess|ess_remote, got '%s'",
                    depth_source_.c_str());
                return false;
            }
            usb_driver_ = std::make_unique<core::UsbStereoDriver>();
            if (!usb_driver_->open(usb_cfg)) {
                RCLCPP_ERROR(
                    get_logger(), "Failed to open USB stereo camera: %s",
                    usb_driver_->lastError().c_str());
                return false;
            }

            if (depth_source_ == "ess_remote") {
                // [ess_remote] 双机模式：本机不跑 ESS 模型，
                // 建立远程链路（发拼接图、发标定、收远程深度）
                pub_remote_stereo_ = this->create_publisher<sensor_msgs::msg::Image>(
                    remote_stereo_topic_, rclcpp::SensorDataQoS().keep_last(1));
                // 注意：必须用 volatile（默认）。进程内通信(intra-process)不允许 latched，
                // launch 里 fusion_node 开了 use_intra_process_comms=True，transient_local 会抛异常。
                // 标定改为每帧携带，保证晚启动的服务端也能收到（见 run_ess_depth_remote）。
                pub_remote_calib_ = this->create_publisher<wheel_msgs::msg::StereoCalibration>(
                    remote_calib_topic_, rclcpp::QoS(1));
                sub_remote_depth_ = this->create_subscription<sensor_msgs::msg::Image>(
                    remote_depth_topic_, rclcpp::SensorDataQoS().keep_last(1),
                    std::bind(&FusionNode::remote_depth_callback, this,
                              std::placeholders::_1));
                RCLCPP_INFO(
                    get_logger(),
                    "Camera mode: USB stereo + REMOTE ESS depth (%s -> %s, "
                    "publish %dms, stale timeout %lldms)",
                    remote_stereo_topic_.c_str(), remote_depth_topic_.c_str(),
                    remote_publish_interval_ms_,
                    static_cast<long long>(remote_stale_timeout_ms_));
                return true;
            }

            ess_engine_ = std::make_unique<core::EssDepthEngine>(ess_cfg);
            if (!ess_engine_->initialize()) {
                RCLCPP_ERROR(get_logger(), "Failed to initialize ESS depth engine");
                return false;
            }
            RCLCPP_INFO(
                get_logger(), "Camera mode: USB stereo + ESS depth (%dx%d -> ESS %dx%d)",
                usb_cfg.output_width, usb_cfg.output_height,
                ess_engine_->inputWidth(), ess_engine_->inputHeight());
            return true;
        }

        if (camera_type_ != "zed") {
            RCLCPP_ERROR(get_logger(), "Unsupported camera.type: '%s'", camera_type_.c_str());
            return false;
        }
        driver_ = std::make_unique<core::ZedDriver>();
        return driver_->open(zed_cfg);
    }

    bool run_ess_depth(core::ZedGpuFrame& frame) {
        // [ess_remote] 双机模式：本地不推理，走远程链路
        if (depth_source_ == "ess_remote") {
            return run_ess_depth_remote(frame);
        }
        if (!ess_engine_) {
            RCLCPP_ERROR(get_logger(), "ESS depth engine is not initialized");
            return false;
        }
        if (frame.rgb_ptr_dev == nullptr || frame.right_ptr_dev == nullptr ||
            frame.rgb_step_bytes <= 0 || frame.right_step_bytes <= 0) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "USB frame is missing left/right BGRA GPU buffers; dropping frame");
            return false;
        }

        const auto params = usb_driver_->getCameraParams();
        core::EssDepthEngine::StereoCalibration calibration;
        calibration.fx = params.fx;
        calibration.fy = params.fy;
        calibration.cx = params.cx;
        calibration.cy = params.cy;
        calibration.baseline_m = params.baseline;

        core::EssDepthEngine::InferenceOutput output;
        if (!ess_engine_->infer(
                frame.rgb_ptr_dev, static_cast<std::size_t>(frame.rgb_step_bytes),
                frame.right_ptr_dev, static_cast<std::size_t>(frame.right_step_bytes),
                frame.width, frame.height, calibration, output)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "ESS depth inference failed; dropping frame");
            return false;
        }
        if (!ess_engine_->waitForCompletion()) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "ESS depth synchronization failed; dropping frame");
            return false;
        }

        frame.depth_ptr_dev = const_cast<float*>(output.depth_ptr_dev);
        frame.depth_step_bytes = output.width * static_cast<int>(sizeof(float));
        frame.cloud_ptr_dev = const_cast<float4*>(output.cloud_ptr_dev);
        frame.cloud_step_bytes = output.width * static_cast<int>(sizeof(float4));
        return true;
    }

    // [ess_remote] 双机远程深度链路：发拼接图 -> 收远程深度 -> 本地重建点云
    //
    // 返回 true 但不填 cloud_ptr_dev 时，下游 has_depth_input=false 走安全分支
    // （跳过点云融合、AI 语义分割照常）——远程深度未就绪/过期时帧不丢。
    bool run_ess_depth_remote(core::ZedGpuFrame& frame) {
        const int64_t now_ns = static_cast<int64_t>(frame.timestamp_ns);

        // ---- 1. 节流发布拼接图（首次同时发布标定，latched）----
        if (now_ns - last_remote_stereo_pub_ns_ >=
            static_cast<int64_t>(remote_publish_interval_ms_) * 1000000LL) {
            cv::Mat left_bgr, right_bgr;
            usb_driver_->retrieveStereoPair(left_bgr, right_bgr);
            if (!left_bgr.empty() && !right_bgr.empty() &&
                left_bgr.cols == right_bgr.cols && left_bgr.rows == right_bgr.rows) {
                // 标定：每次发图都携带（仅 5 个 float，开销可忽略）。
                // volatile QoS 下不依赖 latched，服务端即使晚启动也会在下一帧收到。
                const auto p = usb_driver_->getCameraParams();
                auto calib_msg = std::make_unique<wheel_msgs::msg::StereoCalibration>();
                calib_msg->header.stamp = rclcpp::Time(now_ns);
                calib_msg->fx = p.fx;
                calib_msg->fy = p.fy;
                calib_msg->cx = p.cx;
                calib_msg->cy = p.cy;
                calib_msg->baseline_m = p.baseline;
                pub_remote_calib_->publish(std::move(calib_msg));
                if (!remote_calib_published_) {
                    remote_calib_published_ = true;
                    RCLCPP_INFO(get_logger(),
                                "[ess_remote] calibration publishing (per-frame): fx=%.2f "
                                "fy=%.2f cx=%.2f cy=%.2f baseline=%.4f m",
                                p.fx, p.fy, p.cx, p.cy, p.baseline);
                }

                // 左右拼接成单张 2560x720，JPEG 压缩后单话题传输（保证左右配对）
                cv::Mat stereo;
                cv::hconcat(left_bgr, right_bgr, stereo);
                std::vector<uchar> jpg;
                std::vector<int> enc_params = {
                    cv::IMWRITE_JPEG_QUALITY, remote_jpeg_quality_};
                cv::imencode(".jpg", stereo, jpg, enc_params);

                auto img_msg = std::make_unique<sensor_msgs::msg::Image>();
                img_msg->header.stamp = rclcpp::Time(now_ns);
                img_msg->header.frame_id = "zed_left_camera_frame";
                img_msg->height = static_cast<uint32_t>(stereo.rows);
                img_msg->width = static_cast<uint32_t>(stereo.cols);
                img_msg->encoding = "jpeg";
                img_msg->is_bigendian = false;
                img_msg->step = static_cast<uint32_t>(jpg.size());
                img_msg->data.assign(jpg.begin(), jpg.end());
                pub_remote_stereo_->publish(std::move(img_msg));
                last_remote_stereo_pub_ns_ = now_ns;
            }
        }

        // ---- 2. 取最新远程深度 ----
        sensor_msgs::msg::Image::SharedPtr depth_msg;
        {
            std::lock_guard<std::mutex> lock(frame_mutex_);
            depth_msg = latest_remote_depth_;
        }
        frame.depth_ptr_dev = nullptr;
        frame.cloud_ptr_dev = nullptr;
        frame.depth_step_bytes = 0;
        frame.cloud_step_bytes = 0;

        if (!depth_msg) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "[ess_remote] no remote depth yet; waiting for %s",
                                 remote_depth_topic_.c_str());
            return true;
        }
        const int64_t depth_stamp_ns =
            static_cast<int64_t>(depth_msg->header.stamp.sec) * 1000000000LL +
            static_cast<int64_t>(depth_msg->header.stamp.nanosec);
        const int64_t age_ms = (now_ns - depth_stamp_ns) / 1000000LL;
        if (age_ms > remote_stale_timeout_ms_ || age_ms < -100) {
            if (!reuse_stale_depth_) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 2000,
                    "[ess_remote] remote depth stale (age %lld ms, limit %lld ms); "
                    "skipping depth fusion this frame",
                    static_cast<long long>(age_ms),
                    static_cast<long long>(remote_stale_timeout_ms_));
                return true;
            }
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 5000,
                "[ess_remote] remote depth stale (age %lld ms, limit %lld ms); "
                "reusing last depth",
                static_cast<long long>(age_ms),
                static_cast<long long>(remote_stale_timeout_ms_));
        }

        // ---- 3. 格式/尺寸校验 ----
        const int img_w = frame.width, img_h = frame.height;
        if (depth_msg->encoding != "16UC1" ||
            depth_msg->width != static_cast<uint32_t>(img_w) ||
            depth_msg->height != static_cast<uint32_t>(img_h)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "[ess_remote] depth format mismatch (%s %ux%u, want 16UC1 %dx%d)",
                depth_msg->encoding.c_str(), depth_msg->width, depth_msg->height,
                img_w, img_h);
            return true;
        }

        // ---- 4. 16UC1 毫米 -> float4 点云（复刻 dataset BLOCK 1 模式）----
        const auto params = usb_driver_->getCameraParams();
        float fx = 527.3f, fy = 527.3f, cx = img_w / 2.0f, cy = img_h / 2.0f;
        if (params.fx > 10.0f) {
            fx = params.fx; fy = params.fy; cx = params.cx; cy = params.cy;
        }
        const uint16_t* depth_u =
            reinterpret_cast<const uint16_t*>(depth_msg->data.data());

        size_t cloud_bytes = static_cast<size_t>(img_w) * img_h * sizeof(float4);
        if (d_cloud_buffer_size_ < cloud_bytes) {
            if (d_cloud_buffer_) cudaFree(d_cloud_buffer_);
            if (cudaMalloc(&d_cloud_buffer_, cloud_bytes) != cudaSuccess) {
                RCLCPP_ERROR(get_logger(), "[ess_remote] cudaMalloc cloud buffer failed");
                return true;
            }
            d_cloud_buffer_size_ = cloud_bytes;
        }

        std::vector<float4> host_cloud(static_cast<size_t>(img_w) * img_h);
        // 在 Z_UP_X_FWD 坐标系下：X向前, Y向左, Z向上（与 BLOCK 1 一致）
        for (int v = 0; v < img_h; ++v) {
            for (int u = 0; u < img_w; ++u) {
                const float d = depth_u[v * img_w + u] * 0.001f;
                float4 pt;
                if (d > 0.1f && d < 20.0f && std::isfinite(d)) {
                    pt.x = d;
                    pt.y = d * (cx - u) / fx;
                    pt.z = d * (cy - v) / fy;
                    pt.w = 0.0f;
                } else {
                    pt.x = 0; pt.y = 0; pt.z = 0; pt.w = 0;
                }
                host_cloud[static_cast<size_t>(v) * img_w + u] = pt;
            }
        }
        if (cudaMemcpy(d_cloud_buffer_, host_cloud.data(), cloud_bytes,
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            RCLCPP_ERROR(get_logger(), "[ess_remote] cudaMemcpy cloud buffer failed");
            return true;
        }
        frame.cloud_ptr_dev = reinterpret_cast<float4*>(d_cloud_buffer_);
        frame.cloud_step_bytes = img_w * static_cast<int>(sizeof(float4));
        return true;
    }

    // [ess_remote] 远程深度回传回调（只在锁内存指针，update_loop 取用）
    void remote_depth_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_remote_depth_ = msg;
    }

    bool init_ai(const std::string& p) {
        core::AiEngine::Config cfg; cfg.engine_path = p;
        ai_engine_ = std::make_unique<core::AiEngine>(cfg);
        return ai_engine_->initialize();
    }

    // [ess_remote] 远程 ESS 深度链路成员
    std::string remote_stereo_topic_ = "ess_remote/stereo_input";
    std::string remote_depth_topic_ = "ess_remote/depth";
    std::string remote_calib_topic_ = "ess_remote/calibration";
    int remote_publish_interval_ms_ = 33;
    int remote_jpeg_quality_ = 85;
    int64_t remote_stale_timeout_ms_ = 500;
    bool reuse_stale_depth_ = true; // 过期深度是否沿用（默认沿用）
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr pub_remote_stereo_;
    rclcpp_lifecycle::LifecyclePublisher<wheel_msgs::msg::StereoCalibration>::SharedPtr pub_remote_calib_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_remote_depth_;
    sensor_msgs::msg::Image::SharedPtr latest_remote_depth_;
    bool remote_calib_published_ = false;
    int64_t last_remote_stereo_pub_ns_ = 0;

    std::string camera_type_ = "zed";
    std::string depth_source_ = "sdk";
    std::unique_ptr<core::ZedDriver> driver_;
    std::unique_ptr<core::UsbStereoDriver> usb_driver_;
    std::unique_ptr<core::EssDepthEngine> ess_engine_;
    std::unique_ptr<core::AiEngine> ai_engine_;
    std::unique_ptr<core::ObstacleFusion> fusion_;
    rclcpp_lifecycle::LifecyclePublisher<wheel_msgs::msg::PerceptionOutput>::SharedPtr pub_perception_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr pub_viz_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr pub_viz_mouse_left_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::Image>::SharedPtr pub_bev_fused_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_;
    rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
    rclcpp::TimerBase::SharedPtr timer_;
    // 1. 调试用的点云发布者 (Lifecycle 类型)
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_rect_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_ellipse_;

    // 2. 临时容器 (用于存放拆分后的点云数据)
    std::vector<float4> cloud_rect_vec_;    // 列表1：右侧边界
    std::vector<float4> cloud_ellipse_vec_; // 列表2：危险障碍物

};

RCLCPP_COMPONENTS_REGISTER_NODE(FusionNode)

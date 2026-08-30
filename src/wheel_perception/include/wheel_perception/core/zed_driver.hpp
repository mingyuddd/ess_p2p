#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
// 前向声明 CUDA 流类型
#include <opencv2/opencv.hpp> // 引用 OpenCV
#include <sensor_msgs/msg/point_cloud2.hpp>

extern bool flag_odom,flag_obstacle;      // 控制里程计采集
extern double current_x_,current_y_,current_z_,
       current_roll_,  // 横滚角 (rad)
       current_pitch_,  // 俯仰角 (rad)
       current_yaw_;  // 航向角 (rad)，绕Z轴，左转为正
extern double x_min,x_last,y_min,y_last,min_d,odx_last,ody_last,right_right;
extern bool use_dataset_mode_,out_left0,out_right0,out_left1,out_right1;
namespace wheel_perception{
namespace core{

 /**
 * @brief 保存 ZED 相机一帧的 GPU 数据指针
 * 注意：这些指针指向显存 (Device Memory)，不可在 CPU 直接读取
 */
struct ZedGpuFrame {
    uint64_t timestamp_ns;  // 时间戳，纳秒级
    void* rgb_ptr_dev = nullptr;    // RGB 图像数据指针 (uint8_t)
    int rgb_step_bytes = 0;    // RGB 图像每行字节数
    void* right_ptr_dev = nullptr;  // USB/ESS 右目校正图像指针 (BGRA)
    int right_step_bytes = 0;

    void* depth_ptr_dev = nullptr;  // 深度图数据指针 (float)
    int depth_step_bytes = 0;

    void* cloud_ptr_dev = nullptr;  // 点云数据指针 (float * 4)
    int cloud_step_bytes = 0;

    int width = 0;
    int height = 0;
    // [新增] 用于存放位置跟踪(Positional Tracking)的里程计和四元数
    float pose_x = 0.0f;
    float pose_y = 0.0f;
    float pose_z = 0.0f;
    float quat_x = 0.0f;
    float quat_y = 0.0f;
    float quat_z = 0.0f;
    float quat_w = 1.0f;
    bool pose_valid = false;
    bool imu_valid = false;


};

/**
 * @brief ZED SDK 的 Pimpl 封装
 * 负责管理相机生命周期，并将数据直接上传到 GPU 指针供后续模块使用
 */
class ZedDriver {
public:
    struct Config{
        int resolution_id = 2;  // 分辨率 ID (0:VGA, 1:HD720, 2:HD1080, 3:HD2K) or  0:2K, 1:1080, 2:720, 3:VGA ？
        int fps = 60;
        float depth_min = 0.3f;
        float depth_max = 20.0f;
        bool flip_camera = false; // 是否翻转相机图像
        int depth_mode = 1; // 对应 sl::DEPTH_MODE::PERFORMANCE
        int coordinate_system = 3;
        std::string svo_path = ""; // SVO 文件路径，空表示实时相机
    };

    ZedDriver();
    ~ZedDriver();

    ZedDriver(const ZedDriver&) = delete;
    ZedDriver& operator=(const ZedDriver&) = delete;

    bool open(const Config& config);
    void close();

    /**
     * @brief 抓取最新一帧数据
     * 这是一个阻塞调用，直到新的一帧准备好
     * * @param out_frame 输出的帧结构体，包含 GPU 指针
     * @return true if grab successful
     */
    bool grab(ZedGpuFrame& out_frame);


    // 接口 1: CPU 端 (OpenCV)
    void retrieveImage(cv::Mat& out_img);
    void retrievePointCloud(sensor_msgs::msg::PointCloud2& msg, const std::string& frame_id);
    // 接口 2: GPU 端
    // 返回 void* 通用指针，避免在头文件中引入 cuda/zed 类型
    void* getRawImageGPUPtr();


    /**
     * @brief 获取相机参数 (用于重投影或发布 CameraInfo)
     * 这里简化处理，返回关键参数
     */
    struct CameraParams{
        float fx,fy,cx,cy;
        float baseline;
    };
    CameraParams getCameraParams() const;

    /**
     * @brief 检查相机是否打开
     */
    bool isOpened() const;

private:
// Pimpl 惯用法：实现细节隐藏在 struct Impl 中
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
    cudaStream_t stream_;
};


}
}


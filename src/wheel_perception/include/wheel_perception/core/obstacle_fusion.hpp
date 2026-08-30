#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <rclcpp_lifecycle/lifecycle_publisher.hpp>

namespace wheel_perception {
namespace core {

struct SceneLineFitResult {
    bool has_line;
    float raw_dist;
    float raw_yaw;
    float p_far_x, p_far_y, p_far_z;
    float p_near_x, p_near_y, p_near_z;
    float last_valid_y; // 方便 CPU 端判断 is_vertical
};

struct PerceptionParams {
    float min_x = 0.3f;  float max_x = 5.0f;
    float min_y = -3.0f; float max_y = 3.0f; 
    float min_z = -0.2f; float max_z = 2.0f;

    // [新增] 体素大小放这里，方便动态调整
    float voxel_size = 0.05f;

    // 降采样和滤波参数
    int step = 2; 
    float ellipse_x = 6.0f;
    float ellipse_y = 100.0f;
    float ellipse_thres = 50.0f;
};

// 2. [关键修复] 必须把这个结构体定义加回来！
// fusion_node.cpp 的 analyze_scene 函数依赖它
struct Obstacle3DStat {
    int class_id = 0;
    int count = 0;
    float min_x=0, max_x=0;
    float min_y=0, max_y=0;
    float min_z=0, max_z=0;
    float sum_x=0, sum_y=0, sum_z=0; 

    void reset() {
        count = 0;
        sum_x = sum_y = sum_z = 0;
        min_x = min_y = min_z = 0;
    }
};

struct RoadEdgePoint {
    float x, y, z;
    bool valid;
};

class ObstacleFusion {
public:
    ObstacleFusion(int ai_width, int ai_height);
    ~ObstacleFusion();

    int filterCloud(
        const void* zed_cloud_ptr, 
        int width, int height, 
        PerceptionParams params);

    void retrieveFilteredCloud(std::vector<float4>& host_cloud);
    void processSegmentation(void* trt_output_ptr);
    void extractRightEdge(
        const void* zed_cloud_ptr,
        int width, int height,
        PerceptionParams params,
        std::vector<RoadEdgePoint>& output_buffer);
    // 同时提取左右道路边界：右边界用于巡线，左边界仅用于道路宽度估计。
    void extractRoadEdges(
        const void* zed_cloud_ptr,
        int width, int height,
        PerceptionParams params,
        std::vector<RoadEdgePoint>& right_output,
        std::vector<RoadEdgePoint>& left_output);

    void downloadMask(unsigned char* host_mask);

    // { changed code } 新增全 GPU 盲区检测接口
    void processBlindZoneOnGPU(
        const void* d_cloud_ptr,
        int img_w, int img_h,
        float curr_x, float curr_y, float curr_yaw,
        bool& out_left, bool& out_right, float& right_dist,
        std::vector<uint8_t>& out_fused_bev
    );
    bool processRoadEdgeOnGPU(SceneLineFitResult& out_result);

private:
    int ai_width_, ai_height_;

        // 历史缓冲配置（最多保存 5 帧）
    struct BevHistory {
        uint8_t* d_grid = nullptr;
        float x = 0; float y = 0; float yaw = 0;
    };
    std::vector<BevHistory> history_queue_;

    // [新增] 调试用的 Lifecycle 发布者
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_rect_;
    rclcpp_lifecycle::LifecyclePublisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_ellipse_;

    // [新增] 临时容器，用于存放拆分后的点云 (保持和 filtered_pts 一样的类型)
    std::vector<float4> cloud_rect_vec_;    // 列表1：右侧边界
    std::vector<float4> cloud_ellipse_vec_; // 列表2：危险障碍物

    // --- GPU 内存 ---
    float4* d_filtered_cloud_ = nullptr;
    int* d_point_count_ = nullptr;
    
    RoadEdgePoint* d_edge_buffer_ = nullptr;
    RoadEdgePoint* d_left_edge_buffer_ = nullptr;
    unsigned char* d_mask_visualization_ = nullptr;

    // 最大道路连通域 GPU 工作区：构造时一次申请，避免逐帧分配和 CPU 往返。
    int* d_ccl_parents_ = nullptr;
    int* d_component_sizes_ = nullptr;
    unsigned long long* d_largest_component_key_ = nullptr;
    
    // [新增] 自适应体素网格相关
    int* d_voxel_grid_ = nullptr;     // 动态网格数组
    size_t voxel_grid_capacity_ = 0;  // 当前显存申请的大小 (防止频繁 malloc)
    // const float VOXEL_SIZE = 0.05f;   // 体素大小 5cm (可配)
    
    const int MAX_POINTS = 1280 * 720;
};

}
}

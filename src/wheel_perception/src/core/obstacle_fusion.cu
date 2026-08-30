#include "wheel_perception/core/obstacle_fusion.hpp"
#include "wheel_perception/core/mask_postprocess.hpp"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm> // for std::max
#include <cuda_runtime_api.h> // For cudaStream_t


extern "C" void launch_argmax_kernel(
    float* logits_ptr, 
    unsigned char* mask_ptr, 
    int w, int h, int num_classes, 
    cudaStream_t stream
);

// =============================================================================
// [新增]: 单通道 Sigmoid 阈值处理 Kernel
// =============================================================================
__global__ void sigmoid_threshold_kernel(
    const float* __restrict__ logits, 
    unsigned char* __restrict__ mask, 
    int width, int height) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    // 单通道输出，类似于 python 的 mask = (prob > 0.5)
    // 根据需要可以将路面设为 1（如果是路面分割）
    mask[idx] = (logits[idx] > 0.5f) ? 1 : 0; 
}


namespace wheel_perception {
namespace core {


// =============================================================================
// [逻辑复刻] 对应 CPU 的 if (...) continue
// =============================================================================
__device__ bool is_point_valid(float4 pt, PerceptionParams params) {
    // 1. NaN / Inf 检查 (保留)
    if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z)) return false;
    if (isinf(pt.x) || isinf(pt.y) || isinf(pt.z)) return false;

    // 2. 矩形框检查 (保留，这才是 max_x=5.0 生效的地方)
    if (pt.x < params.min_x || pt.x > params.max_x) return false;
    if (pt.y < params.min_y || pt.y > params.max_y) return false;
    if (pt.z < params.min_z || pt.z > params.max_z) return false;

    // 3. 【删除或注释掉这两行】
    // float ellipse_val = params.ellipse_x * pt.x * pt.x + params.ellipse_y * pt.y * pt.y;
    // if (ellipse_val > params.ellipse_thres) return false; 

    return true;
}

__device__ int get_point_category(float4 pt, PerceptionParams params) {
    // 1. 基础有效性检查 (共用)
    if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z)) return 0;
    if (pt.z < params.min_z || pt.z > params.max_z) return 0; // 高度过滤
    if (pt.x < -1.0f) return 0; // 绝对后方盲区

    int category = 0;

    // 2. 判定 A: 矩形逻辑 (用于 LQR 巡线边界)
    // 逻辑: 在 max_x 以内，且在 min_y (例如 -0.4) 的左边(y更大的方向)
    if (pt.x <= params.max_x && pt.y >= params.min_y) {
        category |= 1; // 标记二进制第1位 (001)
    }

    // 3. 判定 B: 椭圆逻辑 (用于紧急避障)
    float ellipse_val = params.ellipse_x * pt.x * pt.x + params.ellipse_y * pt.y * pt.y;
    if (ellipse_val < params.ellipse_thres && pt.y >= -0.3) {
        category |= 2; // 标记二进制第2位 (010)
    }

    return category;
}

// 1. 初始化网格 Kernel
__global__ void clear_voxel_grid_kernel(int* grid, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) grid[idx] = -1; 
}


// 2. 自适应体素滤波 Kernel
__global__ void voxel_filter_kernel(
    const float4* __restrict__ src_cloud,
    float4* dst_cloud,
    int* counter,
    int* voxel_grid, 
    int width, int height,
    PerceptionParams params,
    int max_capacity,
    float voxel_size, 
    int grid_dim_x, int grid_dim_y, int grid_dim_z)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (x % params.step != 0 || y % params.step != 0) return;

    int idx = y * width + x;
    float4 pt = src_cloud[idx];

    // [核心修改] 获取分类标记
    int cat = get_point_category(pt, params);
    
    // 如果完全不属于任何一类，直接丢弃
    if (cat == 0) return;

    // === 自适应坐标计算 ===
    // 自动将 min_x 映射到索引 0，不再需要手动算 offset
    // idx = (val - min) / size
    int vx = (int)((pt.x - params.min_x) / voxel_size);
    int vy = (int)((pt.y - params.min_y) / voxel_size);
    int vz = (int)((pt.z - params.min_z) / voxel_size);

    if (vx < 0 || vx >= grid_dim_x || vy < 0 || vy >= grid_dim_y || vz < 0 || vz >= grid_dim_z) return;
    int v_idx = vz * (grid_dim_x * grid_dim_y) + vy * grid_dim_x + vx;

    // [修改] 写入数据时，把 cat 存入 .w 字段
    int old_val = atomicCAS(&voxel_grid[v_idx], -1, idx);
    if (old_val == -1) {
        int write_idx = atomicAdd(counter, 1);
        if (write_idx < max_capacity) {
            pt.w = (float)cat; // <--- 关键：把分类结果带回 CPU
            dst_cloud[write_idx] = pt;
        }
    }
}

// =============================================================================
// Kernel: 右侧边缘提取 (必须复用同一套 is_point_valid)
// =============================================================================
__device__ bool is_edge_point_valid(float4 pt)
{
    return !isnan(pt.x) && !isnan(pt.y) && !isnan(pt.z) &&
           !isinf(pt.x) && !isinf(pt.y) && !isinf(pt.z) &&
           pt.x >= 0.1f && pt.x <= 15.0f;
}

__global__ void extract_road_edges_kernel(
    const unsigned char* __restrict__ mask, 
    const float4* __restrict__ cloud, 
    RoadEdgePoint* right_edge_out,
    RoadEdgePoint* left_edge_out,
    int ai_w, int ai_h,
    int zed_w, int zed_h,
    PerceptionParams params)
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= ai_h) return;

    right_edge_out[y].valid = false;
    left_edge_out[y].valid = false;
    
    // 简单的从右向左扫描，得到右侧道路边界
    int row_offset = y * ai_w;
    int right_pixel_x = -1;
    bool found_right = false;

    for (int x = ai_w - 1; x >= 0; --x) {
        if (mask && mask[row_offset + x] == 0) { // 假设0是路
             right_pixel_x = x; found_right = true; break;
        }
    }

    if (found_right) {
        // 映射坐标
        int zx = right_pixel_x * zed_w / ai_w;
        int zy = y * zed_h / ai_h;
        int z_idx = zy * zed_w + zx;
        float4 pt = cloud[z_idx];

        // [重点修复] 以前这里复用了避障的 is_point_valid，导致地面点因为 z < min_z 被误杀掉
        // 现在直接放宽条件，只要不是 NaN、距离车头 0.1 米以上，就判定为有效边缘点
        if (is_edge_point_valid(pt)) // 允许看前方 15 米的路沿
        {
             right_edge_out[y].x = pt.x;
             right_edge_out[y].y = pt.y;
             right_edge_out[y].z = pt.z;
             right_edge_out[y].valid = true;
        } else {
             right_edge_out[y].valid = false;
        }
    }

    // 从左向右扫描，得到左侧道路边界。左侧只用于宽度估计，不参与主巡线。
    int left_pixel_x = -1;
    bool found_left = false;
    for (int x = 0; x < ai_w; ++x) {
        if (mask && mask[row_offset + x] == 0) {
            left_pixel_x = x; found_left = true; break;
        }
    }

    // 道路区域接触图像左边框时，真实左边界位于视野之外。
    // 保留约 1% 的边框保护区，避免把图像裁剪边缘误认为道路左边界。
    const int left_border_margin = max(2, ai_w / 100);
    if (found_left && left_pixel_x <= left_border_margin) {
        found_left = false;
    }

    if (found_left) {
        int zx = left_pixel_x * zed_w / ai_w;
        int zy = y * zed_h / ai_h;
        int z_idx = zy * zed_w + zx;
        float4 pt = cloud[z_idx];

        if (is_edge_point_valid(pt)) {
             left_edge_out[y].x = pt.x;
             left_edge_out[y].y = pt.y;
             left_edge_out[y].z = pt.z;
             left_edge_out[y].valid = true;
        } else {
             left_edge_out[y].valid = false;
        }
    }
}

// =============================================================================
// Host Implementation
// =============================================================================



ObstacleFusion::ObstacleFusion(int ai_width, int ai_height) 
    : ai_width_(ai_width), ai_height_(ai_height)
{
    cudaMalloc(&d_filtered_cloud_, MAX_POINTS * sizeof(float4));
    cudaMalloc(&d_point_count_, sizeof(int));
    cudaMalloc(&d_edge_buffer_, ai_height_ * sizeof(RoadEdgePoint));
    cudaMalloc(&d_left_edge_buffer_, ai_height_ * sizeof(RoadEdgePoint));
    cudaMalloc(&d_mask_visualization_, ai_width_ * ai_height_ * sizeof(unsigned char));

    const size_t mask_pixel_count = static_cast<size_t>(ai_width_) * ai_height_;
    cudaMalloc(&d_ccl_parents_, mask_pixel_count * sizeof(int));
    cudaMalloc(&d_component_sizes_, mask_pixel_count * sizeof(int));
    cudaMalloc(&d_largest_component_key_, sizeof(unsigned long long));
    
    // d_voxel_grid_ 延迟到 process 时根据 params 分配
}

ObstacleFusion::~ObstacleFusion() {
    if (d_filtered_cloud_) cudaFree(d_filtered_cloud_);
    if (d_point_count_) cudaFree(d_point_count_);
    if (d_edge_buffer_) cudaFree(d_edge_buffer_);
    if (d_left_edge_buffer_) cudaFree(d_left_edge_buffer_);
    if (d_mask_visualization_) cudaFree(d_mask_visualization_);
    if (d_ccl_parents_) cudaFree(d_ccl_parents_);
    if (d_component_sizes_) cudaFree(d_component_sizes_);
    if (d_largest_component_key_) cudaFree(d_largest_component_key_);
    if (d_voxel_grid_) cudaFree(d_voxel_grid_);
}

// 执行滤波
int ObstacleFusion::filterCloud(
    const void* zed_cloud_ptr, 
    int width, int height, 
    PerceptionParams params)
{
    // [修改 1] 使用 params.voxel_size 计算 grid 维度
    int grid_dim_x = (int)ceil((params.max_x - params.min_x) / params.voxel_size);
    int grid_dim_y = (int)ceil((params.max_y - params.min_y) / params.voxel_size);
    int grid_dim_z = (int)ceil((params.max_z - params.min_z) / params.voxel_size);

    // 保护一下，防止算出来是0或负数
    grid_dim_x = std::max(1, grid_dim_x);
    grid_dim_y = std::max(1, grid_dim_y);
    grid_dim_z = std::max(1, grid_dim_z);

    size_t needed_voxels = (size_t)grid_dim_x * grid_dim_y * grid_dim_z;

    // 2. 显存管理：如果需要的比现有的多，就重新分配
    if (needed_voxels > voxel_grid_capacity_) {
        if (d_voxel_grid_) cudaFree(d_voxel_grid_);
        
        // 稍微多申请一点 (1.2倍)，防止参数微调时频繁 realloc
        voxel_grid_capacity_ = (size_t)(needed_voxels * 1.2); 
        cudaMalloc(&d_voxel_grid_, voxel_grid_capacity_ * sizeof(int));
        
        // printf("[GPU] Realloc Voxel Grid: %d x %d x %d\n", grid_dim_x, grid_dim_y, grid_dim_z);
    }

    // 3. 清空网格
    int threads_clear = 256;
    int blocks_clear = (needed_voxels + threads_clear - 1) / threads_clear;
    clear_voxel_grid_kernel<<<blocks_clear, threads_clear>>>(d_voxel_grid_, needed_voxels);
    
    cudaMemset(d_point_count_, 0, sizeof(int));

    // 4. 启动滤波 Kernel (传入动态维度)
    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

voxel_filter_kernel<<<grid, block>>>(
        (const float4*)zed_cloud_ptr,
        d_filtered_cloud_, d_point_count_, d_voxel_grid_,
        width, height, params, MAX_POINTS,
        params.voxel_size,  // <--- 这里改成 params.voxel_size
        grid_dim_x, grid_dim_y, grid_dim_z
    );
    
    cudaDeviceSynchronize();

    int valid_count = 0;
    cudaMemcpy(&valid_count, d_point_count_, sizeof(int), cudaMemcpyDeviceToHost);
    return valid_count;
}

// 取回数据 (给 ROS 发消息用)
void ObstacleFusion::retrieveFilteredCloud(std::vector<float4>& host_cloud) {
    int valid_count = 0;
    cudaMemcpy(&valid_count, d_point_count_, sizeof(int), cudaMemcpyDeviceToHost);
    if (valid_count > 0) {
        if (valid_count > MAX_POINTS) valid_count = MAX_POINTS;
        host_cloud.resize(valid_count);
        cudaMemcpy(host_cloud.data(), d_filtered_cloud_, valid_count * sizeof(float4), cudaMemcpyDeviceToHost);
    } else {
        host_cloud.clear();
    }
}

void ObstacleFusion::extractRightEdge(
    const void* zed_cloud_ptr,
    int width, int height,
    PerceptionParams params,
    std::vector<RoadEdgePoint>& output_buffer)
{
    std::vector<RoadEdgePoint> unused_left;
    extractRoadEdges(zed_cloud_ptr, width, height, params, output_buffer, unused_left);
}

void ObstacleFusion::extractRoadEdges(
    const void* zed_cloud_ptr,
    int width, int height,
    PerceptionParams params,
    std::vector<RoadEdgePoint>& right_output,
    std::vector<RoadEdgePoint>& left_output)
{
    int threads = 256;
    int blocks = (ai_height_ + threads - 1) / threads;
    
    extract_road_edges_kernel<<<blocks, threads>>>(
        d_mask_visualization_, 
        (const float4*)zed_cloud_ptr, 
        d_edge_buffer_,
        d_left_edge_buffer_,
        ai_width_, ai_height_, width, height,
        params
    );
    
    right_output.resize(ai_height_);
    left_output.resize(ai_height_);
    cudaMemcpy(right_output.data(), d_edge_buffer_, ai_height_ * sizeof(RoadEdgePoint), cudaMemcpyDeviceToHost);
    cudaMemcpy(left_output.data(), d_left_edge_buffer_, ai_height_ * sizeof(RoadEdgePoint), cudaMemcpyDeviceToHost);
}

void ObstacleFusion::downloadMask(unsigned char* host_mask) {
    if (d_mask_visualization_ && host_mask) {
        cudaMemcpy(host_mask, d_mask_visualization_, ai_width_ * ai_height_, cudaMemcpyDeviceToHost);
    }
}

void ObstacleFusion::processSegmentation(void* trt_output_ptr) {
    if (!trt_output_ptr || !d_mask_visualization_) return;

    // 假设 Cityscapes 是 19 类 (0-18)
    const int NUM_CLASSES = 19;
    // 调用 Kernel: Float Logits (NCHW) -> Uchar Mask (HW)
    // 直接把 AI 算出来的结果写到 d_mask_visualization_ 显存里
    launch_argmax_kernel(
        (float*)trt_output_ptr, 
        d_mask_visualization_, 
        ai_width_, ai_height_, 
        NUM_CLASSES, 
        0 // 使用默认流，因为 update_loop 里是串行的
    );
    
    // dim3 block(32, 32);
    // dim3 grid((ai_width_ + block.x - 1) / block.x, (ai_height_ + block.y - 1) / block.y);
    // sigmoid_threshold_kernel<<<grid, block>>>(
    //     (const float*)trt_output_ptr, 
    //     d_mask_visualization_, 
    //     ai_width_, ai_height_
    // );

    // ArgMax 与最大连通域 kernel 都在默认流中顺序执行，mask 始终驻留 GPU。
    keepLargestRoadComponentOnGpu(
        d_mask_visualization_,
        ai_width_,
        ai_height_,
        d_ccl_parents_,
        d_component_sizes_,
        d_largest_component_key_,
        0,
        255,
        nullptr);
}

// =============================================================================
// BEV 与盲区 GPU 实现
// =============================================================================

__global__ void project_mask_to_bev_kernel(
    const unsigned char* __restrict__ mask,
    const float4* __restrict__ cloud,
    uint8_t* __restrict__ bev_grid,
    int ai_w, int ai_h, int img_w, int img_h,
    float H_cam
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    int av_start = ai_h / 1.5f;
    if (x >= ai_w || y < av_start || y >= ai_h) return;

    uint8_t label = mask[y * ai_w + x];
    int iu = x * img_w / ai_w;
    int iv = y * img_h / ai_h;
    
    float4 pt = cloud[iv * img_w + iu];

    if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z)) return;
    if (pt.x <= 0.1f || pt.x > 4.0f || abs(pt.y) > 2.0f) return;
    if (abs(pt.z - (-H_cam)) > 0.8f) return;

    int gx = (int)(pt.x / 0.05f);
    int gy = (int)((pt.y + 2.0f) / 0.05f);

    if (gx >= 0 && gx < 80 && gy >= 0 && gy < 80) {
        bev_grid[gx * 80 + gy] = label;
    }
}

// [新增] 几何拼接 Kernel
__global__ void merge_history_bev_kernel(
    uint8_t* __restrict__ curr_grid,
    const uint8_t* __restrict__ hist_grid,
    float dx, float dy, float dyaw
) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x; // 当前帧 x
    int cy = blockIdx.y * blockDim.y + threadIdx.y; // 当前帧 y

    if (cx >= 80 || cy >= 80) return;

    // 当前只处理当前帧中未覆盖(255)的区域进行拼接
    if (curr_grid[cx * 80 + cy] != 255) return;

    // 当前局部系下查询这个格子对应的物理坐标
    float X_cur = cx * 0.05f;
    float Y_cur = cy * 0.05f - 2.0f;

    // { changed code: 修正坐标变换公式 }
    // 我们要问：当前系下的 (X_cur, Y_cur)，在历史系下是哪里 (X_hist, Y_hist)？
    // 逆向推导: 
    //   1. 当前局部系 -> 世界坐标系
    //      X_global = X_cur * cos(yaw_c) - Y_cur * sin(yaw_c) + x_c
    //      Y_global = X_cur * sin(yaw_c) + Y_cur * cos(yaw_c) + y_c
    //   2. 世界坐标系 -> 历史局部系
    //      X_hist = (X_global - x_h) * cos(yaw_h) + (Y_global - y_h) * sin(yaw_h)
    //      Y_hist = -(X_global - x_h) * sin(yaw_h) + (Y_global - y_h) * cos(yaw_h)
    //
    // 为了省传参数，我们传进来的是 dx = x_c - x_h, dy = y_c - y_h, dyaw = yaw_c - yaw_h
    // 组合后，当前系变到历史系的公式简化为:
    float cos_dyaw = cos(dyaw);
    float sin_dyaw = sin(dyaw);
    
    // (X_cur * cos(yaw_c) - Y_cur * sin(yaw_c) + dx) * cos(yaw_h) + ... 
    // 简化为：在原点平移 dx, dy 之后，再旋转 -dyaw
    // 注意：dx, dy 是在世界系下的绝对差值，而我们要把它转化到历史系方向上
    // 上述化简比较容易出错，最安全的是在主机传入 sin_c, cos_c, dx, dy 等原始参数。
    // 但是这里给出一个用相对差值最稳妥的近似公式 (假设 dx, dy 是在当前车体系下的投影位移)：
    
    // 假设历史车体相当于把当前车体 往后移了 dx, 取消了 dyaw 的旋转
    float h_X = X_cur * cos_dyaw - Y_cur * sin_dyaw + dx * cos(-dyaw) - dy * sin(-dyaw);
    float h_Y = X_cur * sin_dyaw + Y_cur * cos_dyaw + dx * sin(-dyaw) + dy * cos(-dyaw);

    // 计算对应的历史网格索引
    int hx = (int)(h_X / 0.05f);
    int hy = (int)((h_Y + 2.0f) / 0.05f);

    if (hx >= 0 && hx < 80 && hy >= 0 && hy < 80) {
        uint8_t hist_label = hist_grid[hx * 80 + hy];
        if (hist_label != 255) {
            curr_grid[cx * 80 + cy] = hist_label;
        }
    }
}

// { changed code } 扩展版的逻辑 Kernel，同时处理阻挡统计和右侧距离寻边
__global__ void evaluate_bev_logic_kernel(
    const uint8_t* __restrict__ fused_bev,
    int* __restrict__ counters, // [0]:center_total, [1]:center_obs, [2]:left_free, [3]:right_free
    float* __restrict__ row_dists, // 每行单独记录最远右侧距离
    int* __restrict__ valid_rows   // 每行是否包含有效非道路点
) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 我们让每个线程负责扫描一 {行} (gx)。
    // 假设我们只看前方 60 格 (3.0米)。
    if (gx > 60) return;

    int center_total_l = 0;
    int center_obs_l = 0;
    int left_free_l = 0;
    int right_free_l = 0;
    int gy_mid = 40;

    // A. 负责统计该行的障碍与可用空间
    for (int gy = 0; gy < 80; ++gy) {
        uint8_t label = fused_bev[gx * 80 + gy];
        if (label == 255) continue;

        if (abs(gy - gy_mid) <= 2) {
            center_total_l++;
            if (label != 0) center_obs_l++;
        } else if (label == 0) {
            if (gy > gy_mid) left_free_l++;
            else right_free_l++;
        }
    }

    // 原子加回全局计数器
    if (center_total_l > 0) atomicAdd(&counters[0], center_total_l);
    if (center_obs_l > 0) atomicAdd(&counters[1], center_obs_l);
    if (left_free_l > 0) atomicAdd(&counters[2], left_free_l);
    if (right_free_l > 0) atomicAdd(&counters[3], right_free_l);

    // B. 右侧寻边逻辑复现
    // 只在中央行附近 (gx_start ~ gx_end) 执行距离探测
    int gx_mid_row = 30; // 60 / 2
    int row_offset = max(1, 60 / 10);
    int gx_start = max(0, gx_mid_row - row_offset);
    int gx_end = min(79, gx_mid_row + row_offset);

    if (gx >= gx_start && gx <= gx_end) {
        float current_row_dist = gy_mid * 0.05f;
        bool found = false;
        
        for (int gy = gy_mid; gy >= 0; --gy) {
            uint8_t label = fused_bev[gx * 80 + gy];
            if (label != 0 && label != 255) { 
                current_row_dist = (gy_mid - gy) * 0.05f;
                found = true;
                break;
            }
        }
        
        row_dists[gx] = current_row_dist;
        valid_rows[gx] = 1; // 标记该行进行了统计
    }
}

void ObstacleFusion::processBlindZoneOnGPU(
    const void* d_cloud_ptr, int img_w, int img_h,
    float curr_x, float curr_y, float curr_yaw,
    bool& out_left, bool& out_right, float& right_dist,
    std::vector<uint8_t>& out_fused_bev)
{
    uint8_t* d_curr_grid;
    int* d_counters;
    cudaMalloc(&d_curr_grid, 80 * 80 * sizeof(uint8_t));
    cudaMemset(d_curr_grid, 255, 80 * 80 * sizeof(uint8_t));
    
    cudaMalloc(&d_counters, 4 * sizeof(int));
    cudaMemset(d_counters, 0, 4 * sizeof(int));

        // 1. 获取当前顿 BEV (纯净画面)
    dim3 proj_block(32, 32);
    dim3 proj_grid((ai_width_ + proj_block.x - 1) / proj_block.x, (ai_height_ + proj_block.y - 1) / proj_block.y);
    project_mask_to_bev_kernel<<<proj_grid, proj_block>>>(
        d_mask_visualization_, (const float4*)d_cloud_ptr, d_curr_grid, 
        ai_width_, ai_height_, img_w, img_h, 1.2f 
    );

    // { changed code } 2. 将当前 [纯净版本] 的当前帧存入历史记录 (基于位移除重)
    static float last_record_x = 0.0f;
    static float last_record_y = 0.0f;
    // 当车移动超过 ~0.05m 时才记录新的一帧 (15帧历史则有~0.75m时空跨度)
    if (static_cast<int>(curr_x * 100) != static_cast<int>(last_record_x * 100)) {
        BevHistory new_hist;
        new_hist.x = curr_x; new_hist.y = curr_y; new_hist.yaw = curr_yaw;
        cudaMalloc(&new_hist.d_grid, 80 * 80 * sizeof(uint8_t));
        cudaMemcpy(new_hist.d_grid, d_curr_grid, 80 * 80 * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
        
        history_queue_.push_back(new_hist);
        
        // 我们可以在显存里保存更多帧，比如 15 帧
        if (history_queue_.size() > 15) {
            cudaFree(history_queue_.front().d_grid);
            history_queue_.erase(history_queue_.begin());
        }
        
        last_record_x = curr_x;
        last_record_y = curr_y;
    }

    // 3. 依次融合历史帧到 d_curr_grid 做后续判断
    dim3 merge_blk(16, 16);
    dim3 merge_grd((80 + merge_blk.x - 1) / merge_blk.x, (80 + merge_blk.y - 1) / merge_blk.y);
    
    for (const auto& hist : history_queue_) {
        float dx = curr_x - hist.x;
        float dy = curr_y - hist.y;
        float dyaw = curr_yaw - hist.yaw;
        merge_history_bev_kernel<<<merge_grd, merge_blk>>>(d_curr_grid, hist.d_grid, dx, dy, dyaw);
    }

    // 4. 新分配用于统计寻边的显存
    float* d_row_dists;
    int* d_valid_rows;
    cudaMalloc(&d_row_dists, 80 * sizeof(float));   // 80行
    cudaMalloc(&d_valid_rows, 80 * sizeof(int));
    cudaMemset(d_row_dists, 0, 80 * sizeof(float));
    cudaMemset(d_valid_rows, 0, 80 * sizeof(int));

    // 每个线程扫描一行 (最大 80 行)
    int threads = 128;
    int blocks = (80 + threads - 1) / threads;
    evaluate_bev_logic_kernel<<<blocks, threads>>>(d_curr_grid, d_counters, d_row_dists, d_valid_rows);

    // 5. 拉回结果
    int h_counters[4];
    float h_row_dists[80];
    int h_valid_rows[80];
    
    cudaMemcpy(h_counters, d_counters, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_row_dists, d_row_dists, 80 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid_rows, d_valid_rows, 80 * sizeof(int), cudaMemcpyDeviceToHost);

    out_fused_bev.resize(80 * 80);
    cudaMemcpy(out_fused_bev.data(), d_curr_grid, 80 * 80 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // 计算阻挡
    int center_total = h_counters[0]; 
    int center_obs = h_counters[1];
    int left_free = h_counters[2];    
    int right_free = h_counters[3];

    out_left = false; 
    out_right = false;
    
    if (center_total > 0 && (float)center_obs / center_total > 0.6f) {
        if (left_free > right_free) out_left = true;
        else out_right = true;
    }

    // 计算右侧平均距离
    float sum_right_dist = 0.0f;
    int rows_count = 0;
    for (int i = 0; i < 80; i++) {
        if (h_valid_rows[i] == 1) {
            sum_right_dist += h_row_dists[i];
            rows_count++;
        }
    }
    right_dist = (rows_count > 0) ? (sum_right_dist / rows_count) : (40 * 0.05f);

    cudaFree(d_curr_grid);
    cudaFree(d_counters);
    cudaFree(d_row_dists);
    cudaFree(d_valid_rows);
}

}
}

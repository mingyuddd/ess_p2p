#pragma once

#include <cuda_runtime_api.h>

namespace wheel_perception {
namespace core {

// 在 GPU 上对 road_label 做 8 邻域连通域筛选，只保留面积最大的道路组件。
// parents、component_sizes 和 largest_component_key 均为调用者预分配的设备内存。
void keepLargestRoadComponentOnGpu(
    unsigned char* device_mask,
    int width,
    int height,
    int* parents,
    int* component_sizes,
    unsigned long long* largest_component_key,
    unsigned char road_label = 0,
    unsigned char non_road_label = 255,
    cudaStream_t stream = nullptr);

}  // namespace core
}  // namespace wheel_perception

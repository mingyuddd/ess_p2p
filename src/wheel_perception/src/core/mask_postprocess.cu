#include "wheel_perception/core/mask_postprocess.hpp"

#include <cuda_runtime.h>

namespace wheel_perception {
namespace core {
namespace {

constexpr int kThreadsPerBlock = 256;

__device__ int findRoot(int* parents, int node)
{
    int parent = parents[node];
    while (parent != node) {
        const int grandparent = parents[parent];
        if (grandparent != parent) {
            // 合并阶段顺便做路径折半，限制并查集父链长度。
            atomicCAS(&parents[node], parent, grandparent);
        }
        node = parent;
        parent = grandparent;
    }
    return node;
}

// 始终把较大的根挂到较小的根，避免并发合并形成环。
__device__ void unionComponents(int* parents, int first, int second)
{
    while (true) {
        const int first_root = findRoot(parents, first);
        const int second_root = findRoot(parents, second);
        if (first_root == second_root) return;

        const int low_root = min(first_root, second_root);
        const int high_root = max(first_root, second_root);
        const int previous = atomicCAS(&parents[high_root], high_root, low_root);
        if (previous == high_root) return;
    }
}

__global__ void initializeLabelsKernel(
    const unsigned char* mask,
    int* parents,
    int pixel_count,
    unsigned char road_label)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;
    parents[index] = mask[index] == road_label ? index : -1;
}

__global__ void mergeRoadNeighborsKernel(
    const unsigned char* mask,
    int* parents,
    int width,
    int height,
    unsigned char road_label)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = width * height;
    if (index >= pixel_count || mask[index] != road_label) return;

    const int x = index % width;
    const int y = index / width;

    // 每条无向边只处理一次；左上、上、右上、左覆盖完整 8 邻域。
    if (x > 0 && mask[index - 1] == road_label) {
        unionComponents(parents, index, index - 1);
    }
    if (y > 0) {
        const int upper = index - width;
        if (mask[upper] == road_label) {
            unionComponents(parents, index, upper);
        }
        if (x > 0 && mask[upper - 1] == road_label) {
            unionComponents(parents, index, upper - 1);
        }
        if (x + 1 < width && mask[upper + 1] == road_label) {
            unionComponents(parents, index, upper + 1);
        }
    }
}

__global__ void compressLabelsStepKernel(int* parents, int pixel_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count || parents[index] < 0) return;

    const int parent = parents[index];
    const int grandparent = parents[parent];
    if (grandparent >= 0 && parent != grandparent) {
        parents[index] = grandparent;
    }
}

__global__ void countComponentAreasKernel(
    const int* parents,
    int* component_sizes,
    int pixel_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count || parents[index] < 0) return;
    atomicAdd(&component_sizes[parents[index]], 1);
}

__global__ void findLargestComponentKernel(
    const int* parents,
    const int* component_sizes,
    unsigned long long* largest_component_key,
    int pixel_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count || parents[index] != index) return;

    const unsigned int area = static_cast<unsigned int>(component_sizes[index]);
    if (area == 0) return;

    // 高 32 位按面积比较；面积相同时优先保留扫描顺序更早（根索引更小）的组件。
    const unsigned long long tie_breaker = 0xFFFFFFFFu - static_cast<unsigned int>(index);
    const unsigned long long key =
        (static_cast<unsigned long long>(area) << 32) | tie_breaker;
    atomicMax(largest_component_key, key);
}

__global__ void filterLargestComponentKernel(
    unsigned char* mask,
    const int* parents,
    const unsigned long long* largest_component_key,
    int pixel_count,
    unsigned char road_label,
    unsigned char non_road_label)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count || mask[index] != road_label) return;

    const unsigned int encoded_root =
        static_cast<unsigned int>(*largest_component_key & 0xFFFFFFFFu);
    const int largest_root = static_cast<int>(0xFFFFFFFFu - encoded_root);
    if (parents[index] != largest_root) {
        mask[index] = non_road_label;
    }
}

}  // namespace

void keepLargestRoadComponentOnGpu(
    unsigned char* device_mask,
    int width,
    int height,
    int* parents,
    int* component_sizes,
    unsigned long long* largest_component_key,
    unsigned char road_label,
    unsigned char non_road_label,
    cudaStream_t stream)
{
    if (!device_mask || !parents || !component_sizes || !largest_component_key ||
        width <= 0 || height <= 0) {
        return;
    }

    const int pixel_count = width * height;
    const int blocks = (pixel_count + kThreadsPerBlock - 1) / kThreadsPerBlock;

    initializeLabelsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        device_mask, parents, pixel_count, road_label);
    mergeRoadNeighborsKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        device_mask, parents, width, height, road_label);

    // 每轮把父指针跳到祖父；ceil(log2(pixel_count)) 轮可压平最坏长度的父链。
    for (int span = 1; span < pixel_count; span <<= 1) {
        compressLabelsStepKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
            parents, pixel_count);
    }

    cudaMemsetAsync(component_sizes, 0, pixel_count * sizeof(int), stream);
    countComponentAreasKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        parents, component_sizes, pixel_count);

    cudaMemsetAsync(largest_component_key, 0, sizeof(unsigned long long), stream);
    findLargestComponentKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        parents, component_sizes, largest_component_key, pixel_count);
    filterLargestComponentKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        device_mask,
        parents,
        largest_component_key,
        pixel_count,
        road_label,
        non_road_label);
}

}  // namespace core
}  // namespace wheel_perception

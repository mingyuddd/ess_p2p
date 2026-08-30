#include <cuda_runtime.h>

#include <cstddef>

namespace {

__device__ __forceinline__ float clampFloat(float value, float minimum, float maximum) {
    return fminf(fmaxf(value, minimum), maximum);
}

__device__ __forceinline__ uchar4 loadBgra(
    const unsigned char* source, std::size_t step_bytes, int x, int y) {
    const auto* row = reinterpret_cast<const uchar4*>(source +
        static_cast<std::size_t>(y) * step_bytes);
    return row[x];
}

__global__ void essPreprocessBgraKernel(
    const unsigned char* source, std::size_t source_step_bytes,
    int source_width, int source_height, float* destination,
    int destination_width, int destination_height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= destination_width || y >= destination_height) return;

    // Half-pixel bilinear mapping, equivalent to the resize convention used by
    // the NVIDIA ESS preprocessing path.
    const float source_x =
        (static_cast<float>(x) + 0.5F) * static_cast<float>(source_width) /
        static_cast<float>(destination_width) - 0.5F;
    const float source_y =
        (static_cast<float>(y) + 0.5F) * static_cast<float>(source_height) /
        static_cast<float>(destination_height) - 0.5F;

    const float clamped_x = clampFloat(source_x, 0.0F, static_cast<float>(source_width - 1));
    const float clamped_y = clampFloat(source_y, 0.0F, static_cast<float>(source_height - 1));
    const int x0 = static_cast<int>(floorf(clamped_x));
    const int y0 = static_cast<int>(floorf(clamped_y));
    const int x1 = min(x0 + 1, source_width - 1);
    const int y1 = min(y0 + 1, source_height - 1);
    const float wx = clamped_x - static_cast<float>(x0);
    const float wy = clamped_y - static_cast<float>(y0);

    const uchar4 p00 = loadBgra(source, source_step_bytes, x0, y0);
    const uchar4 p10 = loadBgra(source, source_step_bytes, x1, y0);
    const uchar4 p01 = loadBgra(source, source_step_bytes, x0, y1);
    const uchar4 p11 = loadBgra(source, source_step_bytes, x1, y1);

    const float weights[4] = {
        (1.0F - wx) * (1.0F - wy), wx * (1.0F - wy),
        (1.0F - wx) * wy, wx * wy};

    // ZED supplies BGRA. ESS is trained with RGB input. NVIDIA's reference path
    // resizes BGR, swaps R/B, normalizes, then converts interleaved to planar.
    const float blue = weights[0] * p00.x + weights[1] * p10.x +
        weights[2] * p01.x + weights[3] * p11.x;
    const float green = weights[0] * p00.y + weights[1] * p10.y +
        weights[2] * p01.y + weights[3] * p11.y;
    const float red = weights[0] * p00.z + weights[1] * p10.z +
        weights[2] * p01.z + weights[3] * p11.z;

    constexpr float kScale = (1.0F / 255.0F) / 0.5F;
    const int plane_size = destination_width * destination_height;
    const int output_index = y * destination_width + x;
    destination[output_index] = (red - 128.0F) * kScale;
    destination[plane_size + output_index] = (green - 128.0F) * kScale;
    destination[2 * plane_size + output_index] = (blue - 128.0F) * kScale;
}

__global__ void essConfidenceThresholdKernel(
    const float* disparity, const float* confidence, float* filtered_disparity,
    int pixel_count, float threshold) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;
    filtered_disparity[index] =
        (isfinite(confidence[index]) && confidence[index] > threshold) ?
        disparity[index] : -1.0F;
}

}  // namespace

extern "C" cudaError_t launch_ess_preprocess_bgra(
    const void* source_bgra, std::size_t source_step_bytes,
    int source_width, int source_height, float* destination_nchw,
    int destination_width, int destination_height, cudaStream_t stream) {
    if (source_bgra == nullptr || destination_nchw == nullptr || source_width <= 0 ||
        source_height <= 0 || destination_width <= 0 || destination_height <= 0) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(16, 16);
    const dim3 grid(
        (destination_width + block.x - 1) / block.x,
        (destination_height + block.y - 1) / block.y);
    essPreprocessBgraKernel<<<grid, block, 0, stream>>>(
        static_cast<const unsigned char*>(source_bgra), source_step_bytes,
        source_width, source_height, destination_nchw,
        destination_width, destination_height);
    return cudaPeekAtLastError();
}

extern "C" cudaError_t launch_ess_confidence_threshold(
    const float* disparity, const float* confidence, float* filtered_disparity,
    int width, int height, float threshold, cudaStream_t stream) {
    if (disparity == nullptr || confidence == nullptr || filtered_disparity == nullptr ||
        width <= 0 || height <= 0) {
        return cudaErrorInvalidValue;
    }
    const int pixel_count = width * height;
    constexpr int kThreads = 256;
    const int blocks = (pixel_count + kThreads - 1) / kThreads;
    essConfidenceThresholdKernel<<<blocks, kThreads, 0, stream>>>(
        disparity, confidence, filtered_disparity, pixel_count, threshold);
    return cudaPeekAtLastError();
}

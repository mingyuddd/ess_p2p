#include <cuda_runtime.h>

#include <cmath>

namespace {

__device__ __forceinline__ float clampCoordinate(float value, float maximum) {
    return fminf(fmaxf(value, 0.0F), maximum);
}

__global__ void disparityToDepthKernel(
    const float* network_disparity, int network_width, int network_height,
    float* full_disparity, float* depth_m, float4* cloud, int output_width,
    int output_height, float fx, float fy, float cx, float cy, float baseline_m,
    float min_depth_m, float max_depth_m) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= output_width || y >= output_height) return;

    const float source_x =
        (static_cast<float>(x) + 0.5F) * static_cast<float>(network_width) /
        static_cast<float>(output_width) - 0.5F;
    const float source_y =
        (static_cast<float>(y) + 0.5F) * static_cast<float>(network_height) /
        static_cast<float>(output_height) - 0.5F;
    const float clamped_x = clampCoordinate(source_x, static_cast<float>(network_width - 1));
    const float clamped_y = clampCoordinate(source_y, static_cast<float>(network_height - 1));
    const int x0 = static_cast<int>(floorf(clamped_x));
    const int y0 = static_cast<int>(floorf(clamped_y));
    const int x1 = min(x0 + 1, network_width - 1);
    const int y1 = min(y0 + 1, network_height - 1);
    const float wx = clamped_x - static_cast<float>(x0);
    const float wy = clamped_y - static_cast<float>(y0);

    const float d00 = network_disparity[y0 * network_width + x0];
    const float d10 = network_disparity[y0 * network_width + x1];
    const float d01 = network_disparity[y1 * network_width + x0];
    const float d11 = network_disparity[y1 * network_width + x1];
    const float resized_disparity =
        d00 * (1.0F - wx) * (1.0F - wy) + d10 * wx * (1.0F - wy) +
        d01 * (1.0F - wx) * wy + d11 * wx * wy;

    // NVIDIA ESSPostProcessor scales disparity values with the horizontal
    // resolution ratio before/while resizing them to the source image size.
    const float disparity_scale =
        static_cast<float>(output_width) / static_cast<float>(network_width);
    const float disparity_px = resized_disparity * disparity_scale;
    const int output_index = y * output_width + x;
    full_disparity[output_index] = disparity_px;

    float depth_value = nanf("");
    if (isfinite(disparity_px) && disparity_px > 0.0F) {
        const float candidate = fx * baseline_m / disparity_px;
        if (isfinite(candidate) && candidate >= min_depth_m && candidate <= max_depth_m) {
            depth_value = candidate;
        }
    }
    depth_m[output_index] = depth_value;

    if (isfinite(depth_value)) {
        // Match sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD used by ZedDriver:
        // X points forward, Y points left, and Z points up.
        cloud[output_index] = make_float4(
            depth_value,
            depth_value * (cx - static_cast<float>(x)) / fx,
            depth_value * (cy - static_cast<float>(y)) / fy,
            0.0F);
    } else {
        const float invalid = nanf("");
        cloud[output_index] = make_float4(invalid, invalid, invalid, 0.0F);
    }
}

}  // namespace

extern "C" cudaError_t launch_ess_disparity_to_depth_and_cloud(
    const float* network_disparity, int network_width, int network_height,
    float* full_disparity, float* depth_m, float4* cloud, int output_width,
    int output_height, float fx, float fy, float cx, float cy, float baseline_m,
    float min_depth_m, float max_depth_m, cudaStream_t stream) {
    if (network_disparity == nullptr || full_disparity == nullptr || depth_m == nullptr ||
        cloud == nullptr ||
        network_width <= 0 || network_height <= 0 || output_width <= 0 || output_height <= 0 ||
        !std::isfinite(fx) || fx <= 0.0F || !std::isfinite(fy) || fy <= 0.0F ||
        !std::isfinite(cx) || !std::isfinite(cy) ||
        !std::isfinite(baseline_m) || baseline_m <= 0.0F || min_depth_m < 0.0F ||
        max_depth_m <= min_depth_m) {
        return cudaErrorInvalidValue;
    }
    const dim3 block(16, 16);
    const dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y);
    disparityToDepthKernel<<<grid, block, 0, stream>>>(
        network_disparity, network_width, network_height, full_disparity, depth_m, cloud,
        output_width, output_height, fx, fy, cx, cy, baseline_m, min_depth_m, max_depth_m);
    return cudaPeekAtLastError();
}

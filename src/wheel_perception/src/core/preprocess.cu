#include <cuda_runtime.h>
#include <cstdint>

// 声明 Kernel，供 ai_engine.cpp 调用
extern "C" void launch_preprocess_kernel(
    void* src_ptr, int src_w, int src_h, 
    void* dst_ptr, int dst_w, int dst_h, 
    cudaStream_t stream
);

// --------------------------------------------------------------------------
// [FIX 1] 参数修正：必须与 depth_combine.py 完全一致
// Python: Mean=(0.3257, 0.3690, 0.3223), Std=(0.2112, 0.2148, 0.2115)
// 注意：Python顺序是 RGB，这里数组也按 R, G, B 顺序定义
// --------------------------------------------------------------------------
__constant__ float MEAN[3] = {0.3257f, 0.3690f, 0.3223f};
__constant__ float STD[3]  = {0.2112f, 0.2148f, 0.2115f};

// 辅助函数：读取像素并处理边界
__device__ __forceinline__ uchar4 get_pixel(const uchar4* src, int w, int h, int x, int y) {
    x = min(max(x, 0), w - 1);
    y = min(max(y, 0), h - 1);
    return src[y * w + x];
}

__global__ void preprocess_kernel_bilinear(
    const uchar4* __restrict__ src, int src_w, int src_h,
    float* __restrict__ dst, int dst_w, int dst_h) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst_w || y >= dst_h) return;

    // ----------------------------------------------------------------------
    // [FIX 3] 双线性插值 (Bilinear Interpolation)
    // 逻辑复刻 PyTorch 的 align_corners=False
    // ----------------------------------------------------------------------
    float src_x = (x + 0.5f) * ((float)src_w / dst_w) - 0.5f;
    float src_y = (y + 0.5f) * ((float)src_h / dst_h) - 0.5f;

    int x_low = floorf(src_x);
    int y_low = floorf(src_y);
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    float lx = src_x - x_low;
    float ly = src_y - y_low;
    float hx = 1.0f - lx;
    float hy = 1.0f - ly;

    // 读取四个角点 (ZED数据是 BGRA)
    uchar4 p00 = get_pixel(src, src_w, src_h, x_low, y_low);
    uchar4 p10 = get_pixel(src, src_w, src_h, x_high, y_low);
    uchar4 p01 = get_pixel(src, src_w, src_h, x_low, y_high);
    uchar4 p11 = get_pixel(src, src_w, src_h, x_high, y_high);

    // 分通道插值 (x=B, y=G, z=R)
    float b_val = (p00.x * hx + p10.x * lx) * hy + (p00.x * hx + p10.x * lx) * ly; // Bug fixed below
    // 正确的双线性插值公式：
    // Val = (P00 * hx * hy) + (P10 * lx * hy) + (P01 * hx * ly) + (P11 * lx * ly)
    
    // 重新计算权重乘积，避免写错
    float w00 = hx * hy;
    float w10 = lx * hy;
    float w01 = hx * ly;
    float w11 = lx * ly;

    float val_b = p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11;
    float val_g = p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11;
    float val_r = p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11;

    // ----------------------------------------------------------------------
    // [FIX 2] 通道重排与归一化
    // PyTorch 输入要求是 RGB Planar
    // Plane 0 = R, Plane 1 = G, Plane 2 = B
    // 之前的代码把 Blue 放在了 Plane 0，这是错误的！
    // ----------------------------------------------------------------------
    int area = dst_w * dst_h;
    int idx = y * dst_w + x;

    // 归一化公式: (Value/255.0 - Mean) / Std
    dst[idx]            = (val_r / 255.0f - MEAN[0]) / STD[0]; // R @ Plane 0
    dst[idx + area]     = (val_g / 255.0f - MEAN[1]) / STD[1]; // G @ Plane 1
    dst[idx + area * 2] = (val_b / 255.0f - MEAN[2]) / STD[2]; // B @ Plane 2

    // dst[idx]            = val_r / 255.0f; // R @ Plane 0
    // dst[idx + area]     = val_g / 255.0f; // G @ Plane 1
    // dst[idx + area * 2] = val_b / 255.0f; // B @ Plane 2
}

void launch_preprocess_kernel(
    void* src_ptr, int src_w, int src_h, 
    void* dst_ptr, int dst_w, int dst_h, 
    cudaStream_t stream) 
{
    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);

    preprocess_kernel_bilinear<<<grid, block, 0, stream>>>(
        (const uchar4*)src_ptr, src_w, src_h,
        (float*)dst_ptr, dst_w, dst_h
    );
}

// =============================================================================
// [新增] ArgMax Kernel: Logits (Float NCHW) -> Mask (Uchar HW)
// =============================================================================
// 将 AI 输出的 19 层概率图压缩成 1 层类别索引图
// -----------------------------------------------------------------------------
__global__ void argmax_kernel_nchw(
    const float* __restrict__ logits, // Input: 1 x 19 x H x W
    unsigned char* __restrict__ mask, // Output: H x W
    int w, int h, int num_classes) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int area = w * h;
    
    if (idx >= area) return;

    // 寻找最大概率对应的 channel index
    float max_val = -1e30f; // 负无穷
    unsigned char max_idx = 0;

    // 遍历 19 个类别通道 (Memory Layout: Planar)
    // Channel i 的像素 offset = i * area + idx
    for (int c = 0; c < num_classes; ++c) {
        float val = logits[c * area + idx];
        if (val > max_val) {
            max_val = val;
            max_idx = (unsigned char)c;
        }
    }
    mask[idx] = max_idx;
}

// Host 封装
extern "C" void launch_argmax_kernel(
    float* logits_ptr, 
    unsigned char* mask_ptr, 
    int w, int h, int num_classes, 
    cudaStream_t stream)
{
    int total_pixels = w * h;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    
    argmax_kernel_nchw<<<blocks, threads, 0, stream>>>(
        logits_ptr, mask_ptr, w, h, num_classes
    );
}
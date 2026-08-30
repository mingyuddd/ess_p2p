#pragma once
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace wheel_perception {
namespace core {

class AiEngine {
public:
    struct Config {
        std::string engine_path;
        // 其他参数可在此扩展
    };

    AiEngine(const Config& config);
    ~AiEngine();

    bool initialize();

    /**
     * @brief 通用推理接口 (Zero-Copy)
     * @param src_gpu_ptr ZED 提供的原始 GPU 指针 (BGRA / uchar4)
     * @param src_width   ZED 图像宽
     * @param src_height  ZED 图像高
     */
    void infer(void* src_gpu_ptr, int src_width, int src_height);
    
    // 获取输出结果的 GPU 指针
    void* getOutputTensor(const std::string& tensor_name);

    // [新增] 获取模型期望的输入分辨率 (宽, 高)
    void getInputResolution(int& out_w, int& out_h);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace core
} // namespace wheel_perception
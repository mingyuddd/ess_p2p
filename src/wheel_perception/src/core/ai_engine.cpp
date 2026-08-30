#include "wheel_perception/core/ai_engine.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferVersion.h>

// -------------------------------------------------------------------------
// 声明外部定义的 CUDA 启动函数 (来自 preprocess.cu)
// -------------------------------------------------------------------------
extern "C" void launch_preprocess_kernel(
    void* src_ptr, int src_w, int src_h, 
    void* dst_ptr, int dst_w, int dst_h, 
    cudaStream_t stream
);

// Logger 实现
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印警告及以上，屏蔽 Info
        if (severity <= Severity::kWARNING) 
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

namespace wheel_perception {
namespace core {

class AiEngine::Impl {
public:
    Config config_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    // Binding 管理
    int input_idx_ = -1;
    int output_idx_ = -1;
    void* bindings_[2] = {nullptr, nullptr}; // 假设单输入单输出
    std::string input_name_;
    std::string output_name_;
    
    // 维度信息
    nvinfer1::Dims input_dims_;
    
    cudaStream_t stream_ = nullptr;

    Impl(const Config& config) : config_(config) {}

    ~Impl() {
        if (bindings_[0]) cudaFree(bindings_[0]);
        if (bindings_[1]) cudaFree(bindings_[1]);
        if (stream_) cudaStreamDestroy(stream_);
    }

    bool initialize() {
        // 1. 加载模型文件
        std::ifstream file(config_.engine_path, std::ios::binary);
        if (!file.good()) {
            std::cerr << "[AiEngine] Error: Engine file not found: " << config_.engine_path << std::endl;
            return false;
        }
        file.seekg(0, file.end);
        size_t size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> model_stream(size);
        file.read(model_stream.data(), size);
        file.close();

        // 2. 反序列化
        runtime_.reset(nvinfer1::createInferRuntime(gLogger));
        engine_.reset(runtime_->deserializeCudaEngine(model_stream.data(), size));
        if (!engine_) return false;
        context_.reset(engine_->createExecutionContext());

        // 3. 自适应绑定 (Auto-Bind)
#if NV_TENSORRT_MAJOR >= 10
        int nbBindings = engine_->getNbIOTensors();
        for (int i = 0; i < nbBindings; ++i) {
            const char* tensor_name = engine_->getIOTensorName(i);
            if (!tensor_name) continue;

            if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
                input_idx_ = i;
                input_name_ = tensor_name;
                input_dims_ = engine_->getTensorShape(tensor_name);

                // 计算输入 Buffer 大小 (NCHW float32)
                // input_dims_.d[0] 是 batch (1)
                // input_dims_.d[1] 是 channels (3)
                // input_dims_.d[2] 是 height
                // input_dims_.d[3] 是 width
                size_t size = 1 * 3 * input_dims_.d[2] * input_dims_.d[3] * sizeof(float);
                cudaMalloc(&bindings_[input_idx_], size);

                std::cout << "[AiEngine] Input Found: " << input_name_
                          << " Dims: " << input_dims_.d[3] << "x" << input_dims_.d[2] << std::endl;
            } else {
                output_idx_ = i;
                output_name_ = tensor_name;
                nvinfer1::Dims out_dims = engine_->getTensorShape(tensor_name);

                // 计算输出 Buffer 大小
                size_t vol = 1;
                for(int d=0; d<out_dims.nbDims; ++d) vol *= (out_dims.d[d] < 0 ? 1 : out_dims.d[d]);

                cudaMalloc(&bindings_[output_idx_], vol * sizeof(float));

                std::cout << "[AiEngine] Output Found: " << output_name_ << std::endl;
            }
        }
#else
        int nbBindings = engine_->getNbBindings();
        for (int i = 0; i < nbBindings; ++i) {
            if (engine_->bindingIsInput(i)) {
                input_idx_ = i;
                input_name_ = engine_->getBindingName(i);
                input_dims_ = engine_->getBindingDimensions(i);
                
                // 计算输入 Buffer 大小 (NCHW float32)
                // input_dims_.d[0] 是 batch (1)
                // input_dims_.d[1] 是 channels (3)
                // input_dims_.d[2] 是 height
                // input_dims_.d[3] 是 width
                size_t size = 1 * 3 * input_dims_.d[2] * input_dims_.d[3] * sizeof(float);
                cudaMalloc(&bindings_[i], size);
                
                std::cout << "[AiEngine] Input Found: " << input_name_
                          << " Dims: " << input_dims_.d[3] << "x" << input_dims_.d[2] << std::endl;
            } else {
                output_idx_ = i;
                output_name_ = engine_->getBindingName(i);
                nvinfer1::Dims out_dims = engine_->getBindingDimensions(i);
                
                // 计算输出 Buffer 大小
                size_t vol = 1;
                for(int d=0; d<out_dims.nbDims; ++d) vol *= (out_dims.d[d] < 0 ? 1 : out_dims.d[d]);
                
                cudaMalloc(&bindings_[i], vol * sizeof(float)); 
                
                std::cout << "[AiEngine] Output Found: " << output_name_ << std::endl;
            }
        }
#endif
        
        if (input_idx_ < 0 || output_idx_ < 0) {
            std::cerr << "[AiEngine] Error: Could not find valid input/output bindings!" << std::endl;
            return false;
        }

        cudaStreamCreate(&stream_);
#if NV_TENSORRT_MAJOR >= 10
        context_->setTensorAddress(input_name_.c_str(), bindings_[input_idx_]);
        context_->setTensorAddress(output_name_.c_str(), bindings_[output_idx_]);
#endif
        return true;
    }

    void infer(void* zed_raw_ptr, int zed_w, int zed_h) {
        if (!context_) return;

        // 1. GPU 预处理 (Zero-Copy)
        // 调用外部定义的 CUDA Kernel
        launch_preprocess_kernel(
            zed_raw_ptr, zed_w, zed_h, 
            bindings_[input_idx_], input_dims_.d[3], input_dims_.d[2], // dst_w, dst_h
            stream_
        );

        // 2. 异步推理
#if NV_TENSORRT_MAJOR >= 10
        context_->setTensorAddress(input_name_.c_str(), bindings_[input_idx_]);
        context_->setTensorAddress(output_name_.c_str(), bindings_[output_idx_]);
        context_->enqueueV3(stream_);
#else
        context_->enqueueV2(bindings_, stream_, nullptr);
#endif
        
        // 注意：这里不同步，让后续的 Fusion Kernel 继续在 stream_ 上排队执行效率最高
        // 但为了简单起见，且 Fusion 是在另一个 CUDA Stream (默认流) 执行的，
        // 这里需要同步一下，确保 AI 算完了 Fusion 才能读。
        cudaStreamSynchronize(stream_);
    }

    void* getOutputTensor() {
        return bindings_[output_idx_];
    }
    
    // [实现新增接口]
    void getInputResolution(int& w, int& h) {
        // NCHW 格式：0:Batch, 1:Channel, 2:Height, 3:Width
        h = input_dims_.d[2];
        w = input_dims_.d[3];
    }
};

// --- 外壳函数转发 ---
AiEngine::AiEngine(const Config& config) : impl_(std::make_unique<Impl>(config)) {}
AiEngine::~AiEngine() = default;
bool AiEngine::initialize() { return impl_->initialize(); }
void AiEngine::infer(void* ptr, int w, int h) { impl_->infer(ptr, w, h); }
void* AiEngine::getOutputTensor(const std::string&) { return impl_->getOutputTensor(); }

// [转发新增接口]
void AiEngine::getInputResolution(int& w, int& h) { impl_->getInputResolution(w, h); }

} // core
} // wheel_perception

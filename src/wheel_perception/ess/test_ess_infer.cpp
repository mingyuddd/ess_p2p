// test_ess_infer.cpp — 最小化 ESS engine 推理测试（脱离 ROS，定位插件断言失败）
// 用法: test_ess_infer <engine> <plugin> [--fp16-buffer]
// 编译:
//   g++ -std=c++17 -O2 test_ess_infer.cpp \
//       -I/usr/local/cuda-12.2/targets/x86_64-linux/include \
//       -I$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/include \
//       -L/usr/local/cuda-12.2/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -ldl \
//       -L$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib -lcudart \
//       -o test_ess_infer
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace nvinfer1;

class Logger final : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[TRT] " << msg << std::endl;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: test_ess_infer <engine> <plugin>" << std::endl;
        return 2;
    }
    const char* engine_path = argv[1];
    const char* plugin_path = argv[2];

    Logger logger;

    // 1) 插件
    void* handle = dlopen(plugin_path, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) { std::cerr << "dlopen plugin failed: " << dlerror() << std::endl; return 1; }
    std::cout << "[test] plugin loaded: " << plugin_path << std::endl;

    // 2) CUDA 基础检查
    int dev = 0; cudaError_t ce = cudaSetDevice(dev);
    if (ce != cudaSuccess) { std::cerr << "cudaSetDevice: " << cudaGetErrorString(ce) << std::endl; return 1; }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    std::cout << "[test] GPU: " << prop.name << " sm_" << prop.major << prop.minor << std::endl;

    // 3) 读 engine
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f.good()) { std::cerr << "cannot open engine" << std::endl; return 1; }
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> data(sz);
    f.read(data.data(), sz);
    std::cout << "[test] engine bytes: " << sz << std::endl;

    // 4) 反序列化
    std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
    std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(data.data(), sz));
    if (!engine) { std::cerr << "deserialize failed" << std::endl; return 1; }
    std::cout << "[test] engine deserialized" << std::endl;

    std::unique_ptr<IExecutionContext> context(engine->createExecutionContext());
    if (!context) { std::cerr << "context failed" << std::endl; return 1; }
    std::cout << "[test] context created" << std::endl;

    // 5) 打印 I/O
    const int n = engine->getNbIOTensors();
    std::vector<void*> addresses(n, nullptr);
    for (int i = 0; i < n; ++i) {
        const char* name = engine->getIOTensorName(i);
        Dims dims = engine->getTensorShape(name);
        DataType dt = engine->getTensorDataType(name);
        TensorIOMode mode = engine->getTensorIOMode(name);
        std::size_t vol = 1;
        for (int d = 0; d < dims.nbDims; ++d) vol *= dims.d[d];
        std::cout << "[test] " << name << " " << (mode == TensorIOMode::kINPUT ? "in " : "out")
                  << " dims=[";
        for (int d = 0; d < dims.nbDims; ++d) std::cout << (d ? "x" : "") << dims.d[d];
        std::cout << "] " << static_cast<int>(dt) << std::endl;
        if (cudaMalloc(&addresses[i], vol * sizeof(float)) != cudaSuccess) {
            std::cerr << "cudaMalloc failed for " << name << std::endl;
            return 1;
        }
        // 填充输入为随机数据
        if (mode == TensorIOMode::kINPUT) {
            std::vector<float> buf(vol, 0.5f);
            cudaMemcpy(addresses[i], buf.data(), vol * sizeof(float), cudaMemcpyHostToDevice);
        }
        context->setTensorAddress(name, addresses[i]);
    }

    // 6) 推理
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    std::cout << "[test] enqueueV3 ..." << std::endl;
    bool ok = context->enqueueV3(stream);
    if (!ok) {
        std::cerr << "[test] enqueueV3 FAILED" << std::endl;
        return 1;
    }
    cudaError_t sync = cudaStreamSynchronize(stream);
    std::cout << "[test] enqueueV3 OK, sync=" << cudaGetErrorString(sync) << std::endl;

    // 7) 检查输出
    for (int i = 0; i < n; ++i) {
        if (engine->getTensorIOMode(engine->getIOTensorName(i)) == TensorIOMode::kOUTPUT) {
            std::vector<float> out(100);
            cudaMemcpy(out.data(), addresses[i], 100 * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "[test] output[" << engine->getIOTensorName(i) << "] first values: "
                      << out[0] << " " << out[1] << " " << out[2] << std::endl;
        }
    }
    std::cout << "[test] DONE" << std::endl;
    return 0;
}

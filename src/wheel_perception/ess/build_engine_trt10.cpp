// build_engine_trt10.cpp — ONNX → TensorRT engine（TRT 10，替代 trtexec）
// 用法:
//   build_engine_trt10 --onnx=<file.onnx> --saveEngine=<file.engine>
//                      [--plugin=<ess_plugins.so>] [--fp16]
//                      [--input-shapes=name:1x3x288x480;name2:1x3x288x480]
// 编译:
//   g++ -std=c++17 -O2 build_engine_trt10.cpp \
//       -I $TRT10_INC -L $TRT10_LIB -lnvinfer -lnvonnxparser -ldl -o build_engine_trt10
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <dlfcn.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace {

class Logger final : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

void printDims(const char* name, const Dims& d) {
    std::cout << "[build]   " << name << ": [";
    for (int i = 0; i < d.nbDims; ++i) {
        std::cout << (i ? "x" : "") << d.d[i];
    }
    std::cout << "]" << std::endl;
}

// 解析 --input-shapes="name:1x3x288x480;name2:1x3x288x480"
bool parseInputShapes(const std::string& spec, std::map<std::string, Dims>& out) {
    if (spec.empty()) return true;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ';')) {
        const size_t colon = item.find(':');
        if (colon == std::string::npos) return false;
        const std::string name = item.substr(0, colon);
        Dims dims{};
        std::stringstream ds(item.substr(colon + 1));
        std::string token;
        while (std::getline(ds, token, 'x')) {
            if (dims.nbDims >= 8) return false;
            dims.d[dims.nbDims++] = std::atoi(token.c_str());
        }
        out[name] = dims;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string onnx_path, engine_path, plugin_path, input_shapes_spec;
    bool fp16 = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--onnx=", 0) == 0) onnx_path = arg.substr(7);
        else if (arg.rfind("--saveEngine=", 0) == 0) engine_path = arg.substr(13);
        else if (arg.rfind("--plugin=", 0) == 0) plugin_path = arg.substr(9);
        else if (arg.rfind("--input-shapes=", 0) == 0) input_shapes_spec = arg.substr(15);
        else if (arg == "--fp16") fp16 = true;
        else {
            std::cerr << "[build] unknown argument: " << arg << std::endl;
            return 2;
        }
    }
    if (onnx_path.empty() || engine_path.empty()) {
        std::cerr << "usage: build_engine_trt10 --onnx=<file> --saveEngine=<file> "
                     "[--plugin=<so>] [--fp16] [--input-shapes=...]"
                  << std::endl;
        return 2;
    }

    // 1) 加载自定义插件（必须先于 ONNX 解析）
    if (!plugin_path.empty()) {
        void* handle = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!handle) {
            std::cerr << "[build] dlopen plugin failed: " << dlerror() << std::endl;
            return 1;
        }
        std::cout << "[build] plugin loaded: " << plugin_path << std::endl;
    }

    // 2) 解析 ONNX
    Logger logger;
    std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
    if (!builder) {
        std::cerr << "[build] createInferBuilder failed" << std::endl;
        return 1;
    }
    const auto explicit_batch = 1U << static_cast<int>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(explicit_batch));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
        std::cerr << "[build] ONNX parse failed: " << onnx_path << std::endl;
        for (int e = 0; e < parser->getNbErrors(); ++e) {
            std::cerr << "  " << parser->getError(e)->desc() << std::endl;
        }
        return 1;
    }
    std::cout << "[build] parsed ONNX: " << onnx_path << std::endl;

    // 3) 打印输入输出
    std::map<std::string, Dims> shapes;
    if (!parseInputShapes(input_shapes_spec, shapes)) {
        std::cerr << "[build] invalid --input-shapes: " << input_shapes_spec << std::endl;
        return 2;
    }
    std::cout << "[build] network inputs:" << std::endl;
    bool dynamic = false;
    for (int i = 0; i < network->getNbInputs(); ++i) {
        const Dims dims = network->getInput(i)->getDimensions();
        printDims(network->getInput(i)->getName(), dims);
        for (int d = 0; d < dims.nbDims; ++d) {
            if (dims.d[d] < 0) dynamic = true;
        }
    }
    std::cout << "[build] network outputs:" << std::endl;
    for (int i = 0; i < network->getNbOutputs(); ++i) {
        printDims(network->getOutput(i)->getName(), network->getOutput(i)->getDimensions());
    }

    // 4) 构建配置
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 32);  // 4GB workspace
    if (fp16) {
        config->setFlag(BuilderFlag::kFP16);
        std::cout << "[build] precision: FP16" << std::endl;
    }

    // 5) 动态 shape → 需要 optimization profile（min=opt=max=固定值，静态等价）
    if (dynamic) {
        std::cout << "[build] dynamic input shape detected; applying fixed profile" << std::endl;
        // 注意: IOptimizationProfile 析构为 protected，所有权在 addOptimizationProfile 后归 config
        IOptimizationProfile* profile = builder->createOptimizationProfile();
        for (int i = 0; i < network->getNbInputs(); ++i) {
            const char* name = network->getInput(i)->getName();
            Dims dims = network->getInput(i)->getDimensions();
            auto it = shapes.find(name);
            if (it != shapes.end()) {
                dims = it->second;
            } else {
                bool ok = true;
                for (int d = 0; d < dims.nbDims; ++d) {
                    if (dims.d[d] < 0) ok = false;
                }
                if (!ok) {
                    std::cerr << "[build] input '" << name << "' is dynamic but no "
                              << "--input-shapes entry provided" << std::endl;
                    return 1;
                }
            }
            profile->setDimensions(name, OptProfileSelector::kMIN, dims);
            profile->setDimensions(name, OptProfileSelector::kOPT, dims);
            profile->setDimensions(name, OptProfileSelector::kMAX, dims);
            std::cout << "[build]   profile " << name << " = ";
            printDims(name, dims);
        }
        config->addOptimizationProfile(profile);
    }

    // 6) 构建并序列化
    std::cout << "[build] building engine (this may take several minutes)..." << std::endl;
    std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        std::cerr << "[build] engine build failed" << std::endl;
        return 1;
    }

    std::ofstream out(engine_path, std::ios::binary | std::ios::trunc);
    out.write(static_cast<const char*>(serialized->data()), serialized->size());
    out.close();
    std::cout << "[build] engine saved: " << engine_path << " (" << serialized->size()
              << " bytes)" << std::endl;
    return 0;
}

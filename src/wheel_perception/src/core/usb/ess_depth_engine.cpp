#include "wheel_perception/core/ess_depth_engine.hpp"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wheel_perception {
namespace core {
namespace {

constexpr const char* kLeftInputName = "input_left";
constexpr const char* kRightInputName = "input_right";
constexpr const char* kDisparityOutputName = "output_left";
constexpr const char* kConfidenceOutputName = "output_conf";

extern "C" cudaError_t launch_ess_preprocess_bgra(
    const void* source_bgra, std::size_t source_step_bytes,
    int source_width, int source_height, float* destination_nchw,
    int destination_width, int destination_height, cudaStream_t stream);

extern "C" cudaError_t launch_ess_confidence_threshold(
    const float* disparity, const float* confidence, float* filtered_disparity,
    int width, int height, float threshold, cudaStream_t stream);

extern "C" cudaError_t launch_ess_disparity_to_depth_and_cloud(
    const float* network_disparity, int network_width, int network_height,
    float* full_disparity, float* depth_m, float4* cloud, int output_width,
    int output_height, float fx, float fy, float cx, float cy, float baseline_m,
    float min_depth_m, float max_depth_m, cudaStream_t stream);

class EssTensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[EssDepthEngine][TensorRT] " << message << std::endl;
        }
    }
};

EssTensorRtLogger g_ess_logger;

const char* dataTypeName(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF: return "float16";
        case nvinfer1::DataType::kINT8: return "int8";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kBOOL: return "bool";
        case nvinfer1::DataType::kUINT8: return "uint8";
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kFP8: return "fp8";
        case nvinfer1::DataType::kBF16: return "bf16";
        case nvinfer1::DataType::kINT64: return "int64";
        case nvinfer1::DataType::kINT4: return "int4";
#endif
    }
    return "unknown";
}

std::string dimsToString(const nvinfer1::Dims& dims) {
    std::string result = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i != 0) result += "x";
        result += std::to_string(dims.d[i]);
    }
    result += "]";
    return result;
}

bool staticVolume(const nvinfer1::Dims& dims, std::size_t& volume) {
    if (dims.nbDims <= 0) return false;
    volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) return false;
        const auto extent = static_cast<std::size_t>(dims.d[i]);
        if (volume > std::numeric_limits<std::size_t>::max() / extent) return false;
        volume *= extent;
    }
    return true;
}

bool isExpectedInputShape(const nvinfer1::Dims& dims, int height, int width) {
    return dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 3 &&
           dims.d[2] == height && dims.d[3] == width;
}

bool sameDims(const nvinfer1::Dims& lhs, const nvinfer1::Dims& rhs) {
    if (lhs.nbDims != rhs.nbDims) return false;
    for (int i = 0; i < lhs.nbDims; ++i) {
        if (lhs.d[i] != rhs.d[i]) return false;
    }
    return true;
}

bool outputImageSize(const nvinfer1::Dims& dims, int& height, int& width) {
    // ESS engines currently expose [N,H,W]. Accept [N,1,H,W] as well so a
    // regenerated, semantically equivalent engine does not require code changes.
    if (dims.nbDims == 3 && dims.d[0] == 1) {
        height = dims.d[1];
        width = dims.d[2];
        return height > 0 && width > 0;
    }
    if (dims.nbDims == 4 && dims.d[0] == 1 && dims.d[1] == 1) {
        height = dims.d[2];
        width = dims.d[3];
        return height > 0 && width > 0;
    }
    return false;
}

}  // namespace

class EssDepthEngine::Impl {
public:
    struct TensorBuffer {
        std::string name;
        nvinfer1::TensorIOMode mode = nvinfer1::TensorIOMode::kNONE;
        nvinfer1::DataType type = nvinfer1::DataType::kFLOAT;
        nvinfer1::Dims dims{};
        std::size_t bytes = 0;
        void* device_ptr = nullptr;
    };

    explicit Impl(Config config) : config_(std::move(config)) {}

    ~Impl() { reset(); }

    void reset() {
        // TensorRT objects must be destroyed before unloading their plugin code.
        context_.reset();
        engine_.reset();
        runtime_.reset();

        for (auto& tensor : tensors_) {
            if (tensor.device_ptr != nullptr) {
                cudaFree(tensor.device_ptr);
                tensor.device_ptr = nullptr;
            }
        }
        tensors_.clear();

        if (filtered_disparity_ != nullptr) {
            cudaFree(filtered_disparity_);
            filtered_disparity_ = nullptr;
        }
        if (full_disparity_ != nullptr) {
            cudaFree(full_disparity_);
            full_disparity_ = nullptr;
        }
        if (depth_m_ != nullptr) {
            cudaFree(depth_m_);
            depth_m_ = nullptr;
        }
        if (cloud_ != nullptr) {
            cudaFree(cloud_);
            cloud_ = nullptr;
        }
        full_resolution_capacity_ = 0;
        full_width_ = 0;
        full_height_ = 0;
        last_calibration_ = {};

        if (completion_event_ != nullptr) {
            cudaEventDestroy(completion_event_);
            completion_event_ = nullptr;
        }
        if (timing_start_event_ != nullptr) {
            cudaEventDestroy(timing_start_event_);
            timing_start_event_ = nullptr;
        }
        if (preprocess_done_event_ != nullptr) {
            cudaEventDestroy(preprocess_done_event_);
            preprocess_done_event_ = nullptr;
        }
        if (inference_done_event_ != nullptr) {
            cudaEventDestroy(inference_done_event_);
            inference_done_event_ = nullptr;
        }
        if (postprocess_done_event_ != nullptr) {
            cudaEventDestroy(postprocess_done_event_);
            postprocess_done_event_ = nullptr;
        }
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        if (plugin_handle_ != nullptr) {
            dlclose(plugin_handle_);
            plugin_handle_ = nullptr;
        }
        initialized_ = false;
        output_width_ = 0;
        output_height_ = 0;
    }

    bool initialize() {
        if (initialized_) return true;
        reset();

        plugin_handle_ = dlopen(config_.plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (plugin_handle_ == nullptr) {
            std::cerr << "[EssDepthEngine] Failed to load plugin '" << config_.plugin_path
                      << "': " << dlerror() << std::endl;
            reset();
            return false;
        }
        std::cerr << "[EssDepthEngine] plugin loaded: " << config_.plugin_path << std::endl;
        if (!initLibNvInferPlugins(&g_ess_logger, "")) {
            std::cerr << "[EssDepthEngine] Failed to initialize TensorRT plugins" << std::endl;
            reset();
            return false;
        }
        std::cerr << "[EssDepthEngine] initLibNvInferPlugins OK" << std::endl;

        std::ifstream engine_file(config_.engine_path, std::ios::binary | std::ios::ate);
        if (!engine_file.good()) {
            std::cerr << "[EssDepthEngine] Failed to open engine: " << config_.engine_path
                      << std::endl;
            reset();
            return false;
        }
        const std::streamsize engine_size = engine_file.tellg();
        if (engine_size <= 0) {
            std::cerr << "[EssDepthEngine] Engine is empty: " << config_.engine_path << std::endl;
            reset();
            return false;
        }
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(static_cast<std::size_t>(engine_size));
        if (!engine_file.read(engine_data.data(), engine_size)) {
            std::cerr << "[EssDepthEngine] Failed to read complete engine file" << std::endl;
            reset();
            return false;
        }

        runtime_.reset(nvinfer1::createInferRuntime(g_ess_logger));
        if (!runtime_) {
            std::cerr << "[EssDepthEngine] createInferRuntime failed" << std::endl;
            reset();
            return false;
        }
        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if (!engine_) {
            std::cerr << "[EssDepthEngine] Engine deserialization failed" << std::endl;
            reset();
            return false;
        }
        std::cerr << "[EssDepthEngine] engine deserialized (" << engine_size << " bytes)" << std::endl;
        context_.reset(engine_->createExecutionContext());
        if (!context_) {
            std::cerr << "[EssDepthEngine] createExecutionContext failed" << std::endl;
            reset();
            return false;
        }
        std::cerr << "[EssDepthEngine] execution context created" << std::endl;

        if (!inspectAndValidateTensors()) {
            reset();
            return false;
        }
        std::cerr << "[EssDepthEngine] tensors validated" << std::endl;

        cudaError_t cuda_error = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] cudaStreamCreateWithFlags failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            reset();
            return false;
        }
        cuda_error = cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] cudaEventCreateWithFlags failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            reset();
            return false;
        }
        const std::array<cudaEvent_t*, 4> timing_events{
            &timing_start_event_, &preprocess_done_event_, &inference_done_event_,
            &postprocess_done_event_};
        for (cudaEvent_t* event : timing_events) {
            cuda_error = cudaEventCreate(event);
            if (cuda_error != cudaSuccess) {
                std::cerr << "[EssDepthEngine] Timing event creation failed: "
                          << cudaGetErrorString(cuda_error) << std::endl;
                reset();
                return false;
            }
        }

        for (auto& tensor : tensors_) {
            cuda_error = cudaMalloc(&tensor.device_ptr, tensor.bytes);
            if (cuda_error != cudaSuccess) {
                std::cerr << "[EssDepthEngine] cudaMalloc failed for " << tensor.name << " ("
                          << tensor.bytes << " bytes): " << cudaGetErrorString(cuda_error)
                          << std::endl;
                reset();
                return false;
            }
            if (!context_->setTensorAddress(tensor.name.c_str(), tensor.device_ptr)) {
                std::cerr << "[EssDepthEngine] setTensorAddress failed for " << tensor.name
                          << std::endl;
                reset();
                return false;
            }
        }

        const TensorBuffer* disparity = findTensor(kDisparityOutputName);
        if (disparity == nullptr) {
            std::cerr << "[EssDepthEngine] Internal disparity tensor lookup failed" << std::endl;
            reset();
            return false;
        }
        cuda_error = cudaMalloc(
            reinterpret_cast<void**>(&filtered_disparity_), disparity->bytes);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] cudaMalloc failed for filtered disparity: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            reset();
            return false;
        }

        initialized_ = true;
        std::cout << "[EssDepthEngine] Initialized: input=" << config_.input_width << "x"
                  << config_.input_height << " output=" << output_width_ << "x"
                  << output_height_ << " tensors=" << tensors_.size() << std::endl;
        return true;
    }

    bool infer(
        const void* left_bgra_ptr_dev, std::size_t left_step_bytes,
        const void* right_bgra_ptr_dev, std::size_t right_step_bytes,
        int source_width, int source_height, const StereoCalibration& calibration,
        InferenceOutput& output) {
        output = {};
        if (!initialized_ || !context_ || stream_ == nullptr || completion_event_ == nullptr) {
            std::cerr << "[EssDepthEngine] infer called before initialize" << std::endl;
            return false;
        }
        if (left_bgra_ptr_dev == nullptr || right_bgra_ptr_dev == nullptr ||
            source_width <= 0 || source_height <= 0) {
            std::cerr << "[EssDepthEngine] Invalid stereo input pointers or dimensions" << std::endl;
            return false;
        }
        const std::size_t minimum_step = static_cast<std::size_t>(source_width) * 4U;
        if (left_step_bytes < minimum_step || right_step_bytes < minimum_step) {
            std::cerr << "[EssDepthEngine] Invalid stereo input stride: left=" << left_step_bytes
                      << " right=" << right_step_bytes << " minimum=" << minimum_step
                      << std::endl;
            return false;
        }
        if (!std::isfinite(calibration.fx) || calibration.fx <= 0.0F ||
            !std::isfinite(calibration.fy) || calibration.fy <= 0.0F ||
            !std::isfinite(calibration.cx) || !std::isfinite(calibration.cy) ||
            !std::isfinite(calibration.baseline_m) || calibration.baseline_m == 0.0F) {
            std::cerr << "[EssDepthEngine] Invalid stereo calibration: fx=" << calibration.fx
                      << " fy=" << calibration.fy << " cx=" << calibration.cx
                      << " cy=" << calibration.cy
                      << " baseline_m=" << calibration.baseline_m << std::endl;
            return false;
        }
        if (!ensureFullResolutionBuffers(source_width, source_height)) return false;

        TensorBuffer* left_input = findTensorMutable(kLeftInputName);
        TensorBuffer* right_input = findTensorMutable(kRightInputName);
        TensorBuffer* raw_disparity = findTensorMutable(kDisparityOutputName);
        TensorBuffer* confidence = findTensorMutable(kConfidenceOutputName);
        if (!left_input || !right_input || !raw_disparity || !confidence ||
            filtered_disparity_ == nullptr) {
            std::cerr << "[EssDepthEngine] Inference buffers are incomplete" << std::endl;
            return false;
        }

        cudaError_t cuda_error = cudaEventRecord(timing_start_event_, stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Timing start event failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        cuda_error = launch_ess_preprocess_bgra(
            left_bgra_ptr_dev, left_step_bytes, source_width, source_height,
            static_cast<float*>(left_input->device_ptr), config_.input_width,
            config_.input_height, stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Left preprocessing failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        cuda_error = launch_ess_preprocess_bgra(
            right_bgra_ptr_dev, right_step_bytes, source_width, source_height,
            static_cast<float*>(right_input->device_ptr), config_.input_width,
            config_.input_height, stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Right preprocessing failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        cuda_error = cudaEventRecord(preprocess_done_event_, stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Preprocess timing event failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }

        if (!context_->enqueueV3(stream_)) {
            std::cerr << "[EssDepthEngine] TensorRT enqueueV3 failed" << std::endl;
            return false;
        }
        cuda_error = cudaEventRecord(inference_done_event_, stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Inference timing event failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }

        cuda_error = launch_ess_confidence_threshold(
            static_cast<const float*>(raw_disparity->device_ptr),
            static_cast<const float*>(confidence->device_ptr), filtered_disparity_,
            output_width_, output_height_, static_cast<float>(config_.confidence_threshold),
            stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Confidence threshold failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        cuda_error = launch_ess_disparity_to_depth_and_cloud(
            filtered_disparity_, output_width_, output_height_, full_disparity_, depth_m_,
            cloud_, source_width, source_height, calibration.fx, calibration.fy,
            calibration.cx, calibration.cy, std::abs(calibration.baseline_m),
            static_cast<float>(config_.min_depth_m), static_cast<float>(config_.max_depth_m),
            stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Disparity-to-depth conversion failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        cuda_error = cudaEventRecord(postprocess_done_event_, stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Postprocess timing event failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        cuda_error = cudaEventRecord(completion_event_, stream_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] cudaEventRecord failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }

        output.disparity_ptr_dev = full_disparity_;
        output.depth_ptr_dev = depth_m_;
        output.cloud_ptr_dev = cloud_;
        output.confidence_ptr_dev = static_cast<const float*>(confidence->device_ptr);
        output.width = source_width;
        output.height = source_height;
        output.confidence_width = output_width_;
        output.confidence_height = output_height_;
        last_calibration_ = calibration;
        last_calibration_.baseline_m = std::abs(last_calibration_.baseline_m);
        return true;
    }

    bool ensureFullResolutionBuffers(int width, int height) {
        const std::size_t pixel_count =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        if (pixel_count == 0 || pixel_count >
            std::numeric_limits<std::size_t>::max() / sizeof(float4)) {
            std::cerr << "[EssDepthEngine] Invalid full-resolution output dimensions: "
                      << width << "x" << height << std::endl;
            return false;
        }
        if (full_disparity_ != nullptr && depth_m_ != nullptr && cloud_ != nullptr &&
            full_resolution_capacity_ >= pixel_count) {
            full_width_ = width;
            full_height_ = height;
            return true;
        }

        if (full_disparity_ != nullptr) cudaFree(full_disparity_);
        if (depth_m_ != nullptr) cudaFree(depth_m_);
        if (cloud_ != nullptr) cudaFree(cloud_);
        full_disparity_ = nullptr;
        depth_m_ = nullptr;
        cloud_ = nullptr;
        full_resolution_capacity_ = 0;

        const std::size_t bytes = pixel_count * sizeof(float);
        cudaError_t cuda_error = cudaMalloc(
            reinterpret_cast<void**>(&full_disparity_), bytes);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Failed to allocate full disparity: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        cuda_error = cudaMalloc(reinterpret_cast<void**>(&depth_m_), bytes);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Failed to allocate depth: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            cudaFree(full_disparity_);
            full_disparity_ = nullptr;
            return false;
        }
        cuda_error = cudaMalloc(
            reinterpret_cast<void**>(&cloud_), pixel_count * sizeof(float4));
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Failed to allocate ordered point cloud: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            cudaFree(full_disparity_);
            cudaFree(depth_m_);
            full_disparity_ = nullptr;
            depth_m_ = nullptr;
            return false;
        }
        full_resolution_capacity_ = pixel_count;
        full_width_ = width;
        full_height_ = height;
        return true;
    }

    bool getOutputStats(OutputStats& stats) const {
        stats = {};
        if (!initialized_ || completion_event_ == nullptr || full_disparity_ == nullptr ||
            depth_m_ == nullptr || cloud_ == nullptr || full_width_ <= 0 || full_height_ <= 0) {
            return false;
        }
        const TensorBuffer* confidence = findTensor(kConfidenceOutputName);
        if (confidence == nullptr || confidence->device_ptr == nullptr) return false;

        cudaError_t cuda_error = cudaEventSynchronize(completion_event_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Output synchronization failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }

        const std::size_t pixel_count =
            static_cast<std::size_t>(full_width_) * static_cast<std::size_t>(full_height_);
        const std::size_t confidence_pixel_count =
            static_cast<std::size_t>(output_width_) * static_cast<std::size_t>(output_height_);
        std::vector<float> disparity_host(pixel_count);
        std::vector<float> depth_host(pixel_count);
        std::vector<float4> cloud_host(pixel_count);
        std::vector<float> confidence_host(confidence_pixel_count);
        cuda_error = cudaMemcpy(
            disparity_host.data(), full_disparity_, pixel_count * sizeof(float),
            cudaMemcpyDeviceToHost);
        if (cuda_error != cudaSuccess) return false;
        cuda_error = cudaMemcpy(
            depth_host.data(), depth_m_, pixel_count * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_error != cudaSuccess) return false;
        cuda_error = cudaMemcpy(
            cloud_host.data(), cloud_, pixel_count * sizeof(float4), cudaMemcpyDeviceToHost);
        if (cuda_error != cudaSuccess) return false;
        cuda_error = cudaMemcpy(
            confidence_host.data(), confidence->device_ptr,
            confidence_pixel_count * sizeof(float),
            cudaMemcpyDeviceToHost);
        if (cuda_error != cudaSuccess) return false;

        stats.pixel_count = pixel_count;
        stats.min_disparity = std::numeric_limits<float>::infinity();
        stats.max_disparity = -std::numeric_limits<float>::infinity();
        stats.min_confidence = std::numeric_limits<float>::infinity();
        stats.max_confidence = -std::numeric_limits<float>::infinity();
        stats.min_depth_m = std::numeric_limits<float>::infinity();
        stats.max_depth_m = -std::numeric_limits<float>::infinity();
        double disparity_sum = 0.0;
        double depth_sum = 0.0;
        const std::array<std::array<float, 2>, 5> probe_targets{{
            {{last_calibration_.cx, last_calibration_.cy}},
            {{0.25F * static_cast<float>(full_width_), last_calibration_.cy}},
            {{0.75F * static_cast<float>(full_width_), last_calibration_.cy}},
            {{last_calibration_.cx, 0.25F * static_cast<float>(full_height_)}},
            {{last_calibration_.cx, 0.75F * static_cast<float>(full_height_)}}
        }};
        std::array<float, 5> probe_distances{};
        probe_distances.fill(std::numeric_limits<float>::infinity());
        for (std::size_t index = 0; index < pixel_count; ++index) {
            const float disparity_value = disparity_host[index];
            const float depth_value = depth_host[index];
            const float4 point = cloud_host[index];
            if (std::isfinite(disparity_value) && disparity_value > 0.0F) {
                ++stats.valid_disparity_count;
                stats.min_disparity = std::min(stats.min_disparity, disparity_value);
                stats.max_disparity = std::max(stats.max_disparity, disparity_value);
                disparity_sum += disparity_value;
            }
            if (std::isfinite(depth_value) && depth_value > 0.0F) {
                ++stats.valid_depth_count;
                stats.min_depth_m = std::min(stats.min_depth_m, depth_value);
                stats.max_depth_m = std::max(stats.max_depth_m, depth_value);
                depth_sum += depth_value;
            }

            const bool cloud_valid = std::isfinite(point.x) && std::isfinite(point.y) &&
                std::isfinite(point.z) && point.x > 0.0F;
            if (!std::isfinite(depth_value) || depth_value <= 0.0F) {
                if (!std::isnan(point.x) || !std::isnan(point.y) ||
                    !std::isnan(point.z) || point.w != 0.0F) {
                    ++stats.invalid_cloud_encoding_mismatch_count;
                }
                continue;
            }
            if (!cloud_valid) {
                ++stats.cloud_coordinate_mismatch_count;
                continue;
            }

            ++stats.valid_cloud_count;
            const int u = static_cast<int>(index % static_cast<std::size_t>(full_width_));
            const int v = static_cast<int>(index / static_cast<std::size_t>(full_width_));
            const float expected_y = depth_value *
                (last_calibration_.cx - static_cast<float>(u)) / last_calibration_.fx;
            const float expected_z = depth_value *
                (last_calibration_.cy - static_cast<float>(v)) / last_calibration_.fy;
            const float coordinate_error = std::max({
                std::abs(point.x - depth_value), std::abs(point.y - expected_y),
                std::abs(point.z - expected_z), std::abs(point.w)});
            stats.max_cloud_coordinate_error_m =
                std::max(stats.max_cloud_coordinate_error_m, coordinate_error);
            if (coordinate_error > 1.0e-4F) {
                ++stats.cloud_coordinate_mismatch_count;
            }
            const float horizontal_direction =
                last_calibration_.cx - static_cast<float>(u);
            const float vertical_direction =
                last_calibration_.cy - static_cast<float>(v);
            const bool y_direction_wrong = std::abs(horizontal_direction) > 0.5F &&
                point.y * horizontal_direction < 0.0F;
            const bool z_direction_wrong = std::abs(vertical_direction) > 0.5F &&
                point.z * vertical_direction < 0.0F;
            if (y_direction_wrong || z_direction_wrong) {
                ++stats.cloud_direction_mismatch_count;
            }

            for (std::size_t probe_index = 0; probe_index < probe_targets.size();
                 ++probe_index) {
                const float du = static_cast<float>(u) - probe_targets[probe_index][0];
                const float dv = static_cast<float>(v) - probe_targets[probe_index][1];
                const float distance_squared = du * du + dv * dv;
                if (distance_squared < probe_distances[probe_index]) {
                    probe_distances[probe_index] = distance_squared;
                    stats.cloud_probes[probe_index].u = u;
                    stats.cloud_probes[probe_index].v = v;
                    stats.cloud_probes[probe_index].point = point;
                    stats.cloud_probes[probe_index].valid = true;
                }
            }
        }
        for (const float confidence_value : confidence_host) {
            if (std::isfinite(confidence_value)) {
                ++stats.finite_confidence_count;
                stats.min_confidence = std::min(stats.min_confidence, confidence_value);
                stats.max_confidence = std::max(stats.max_confidence, confidence_value);
            }
        }
        if (stats.valid_disparity_count == 0) {
            stats.min_disparity = 0.0F;
            stats.max_disparity = 0.0F;
        } else {
            stats.mean_disparity = static_cast<float>(
                disparity_sum / static_cast<double>(stats.valid_disparity_count));
        }
        if (stats.valid_depth_count == 0) {
            stats.min_depth_m = 0.0F;
            stats.max_depth_m = 0.0F;
        } else {
            stats.mean_depth_m = static_cast<float>(
                depth_sum / static_cast<double>(stats.valid_depth_count));
        }
        if (stats.finite_confidence_count == 0) {
            stats.min_confidence = 0.0F;
            stats.max_confidence = 0.0F;
        }
        return true;
    }

    bool waitForCompletion() const {
        if (!initialized_ || completion_event_ == nullptr) {
            std::cerr << "[EssDepthEngine] Cannot wait before initialization" << std::endl;
            return false;
        }
        const cudaError_t cuda_error = cudaEventSynchronize(completion_event_);
        if (cuda_error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] ESS completion wait failed: "
                      << cudaGetErrorString(cuda_error) << std::endl;
            return false;
        }
        return true;
    }

    bool getTimingStats(TimingStats& stats) const {
        stats = {};
        if (!initialized_ || timing_start_event_ == nullptr ||
            preprocess_done_event_ == nullptr || inference_done_event_ == nullptr ||
            postprocess_done_event_ == nullptr || !waitForCompletion()) {
            return false;
        }
        cudaError_t error = cudaEventElapsedTime(
            &stats.preprocess_ms, timing_start_event_, preprocess_done_event_);
        if (error == cudaSuccess) {
            error = cudaEventElapsedTime(
                &stats.inference_ms, preprocess_done_event_, inference_done_event_);
        }
        if (error == cudaSuccess) {
            error = cudaEventElapsedTime(
                &stats.depth_cloud_ms, inference_done_event_, postprocess_done_event_);
        }
        if (error == cudaSuccess) {
            error = cudaEventElapsedTime(
                &stats.total_ms, timing_start_event_, postprocess_done_event_);
        }
        if (error != cudaSuccess) {
            std::cerr << "[EssDepthEngine] Failed to read GPU timing events: "
                      << cudaGetErrorString(error) << std::endl;
            stats = {};
            return false;
        }
        return true;
    }

    bool inspectAndValidateTensors() {
        const int tensor_count = engine_->getNbIOTensors();
        if (tensor_count != 4) {
            std::cerr << "[EssDepthEngine] Expected exactly 4 I/O tensors, found "
                      << tensor_count << std::endl;
            return false;
        }

        tensors_.reserve(static_cast<std::size_t>(tensor_count));
        for (int index = 0; index < tensor_count; ++index) {
            const char* tensor_name = engine_->getIOTensorName(index);
            if (tensor_name == nullptr) {
                std::cerr << "[EssDepthEngine] Tensor " << index << " has no name" << std::endl;
                return false;
            }

            TensorBuffer tensor;
            tensor.name = tensor_name;
            tensor.mode = engine_->getTensorIOMode(tensor_name);
            tensor.type = engine_->getTensorDataType(tensor_name);
            tensor.dims = engine_->getTensorShape(tensor_name);

            std::size_t volume = 0;
            if (!staticVolume(tensor.dims, volume)) {
                std::cerr << "[EssDepthEngine] Tensor " << tensor.name
                          << " must have static positive dimensions, got "
                          << dimsToString(tensor.dims) << std::endl;
                return false;
            }
            if (tensor.type != nvinfer1::DataType::kFLOAT) {
                std::cerr << "[EssDepthEngine] Tensor " << tensor.name
                          << " must be float32, got " << dataTypeName(tensor.type) << std::endl;
                return false;
            }
            if (volume > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
                std::cerr << "[EssDepthEngine] Tensor " << tensor.name << " is too large"
                          << std::endl;
                return false;
            }
            tensor.bytes = volume * sizeof(float);

            std::cout << "[EssDepthEngine] Tensor " << tensor.name << " mode="
                      << (tensor.mode == nvinfer1::TensorIOMode::kINPUT ? "input" : "output")
                      << " type=" << dataTypeName(tensor.type)
                      << " dims=" << dimsToString(tensor.dims) << std::endl;
            tensors_.push_back(std::move(tensor));
        }

        const TensorBuffer* left = findTensor(kLeftInputName);
        const TensorBuffer* right = findTensor(kRightInputName);
        const TensorBuffer* disparity = findTensor(kDisparityOutputName);
        const TensorBuffer* confidence = findTensor(kConfidenceOutputName);
        if (!left || !right || !disparity || !confidence) {
            std::cerr << "[EssDepthEngine] Required tensor names are: input_left, input_right, "
                         "output_left, output_conf" << std::endl;
            return false;
        }
        if (left->mode != nvinfer1::TensorIOMode::kINPUT ||
            right->mode != nvinfer1::TensorIOMode::kINPUT ||
            disparity->mode != nvinfer1::TensorIOMode::kOUTPUT ||
            confidence->mode != nvinfer1::TensorIOMode::kOUTPUT) {
            std::cerr << "[EssDepthEngine] ESS tensor input/output modes are invalid" << std::endl;
            return false;
        }
        if (!isExpectedInputShape(left->dims, config_.input_height, config_.input_width) ||
            !sameDims(left->dims, right->dims)) {
            std::cerr << "[EssDepthEngine] Expected matching input tensors [1x3x"
                      << config_.input_height << "x" << config_.input_width << "], got left="
                      << dimsToString(left->dims) << " right=" << dimsToString(right->dims)
                      << std::endl;
            return false;
        }
        if (!sameDims(disparity->dims, confidence->dims) ||
            !outputImageSize(disparity->dims, output_height_, output_width_)) {
            std::cerr << "[EssDepthEngine] Disparity/confidence output dimensions are invalid: "
                      << dimsToString(disparity->dims) << " and "
                      << dimsToString(confidence->dims) << std::endl;
            return false;
        }
        return true;
    }

    const TensorBuffer* findTensor(const char* name) const {
        for (const auto& tensor : tensors_) {
            if (tensor.name == name) return &tensor;
        }
        return nullptr;
    }

    TensorBuffer* findTensorMutable(const char* name) {
        for (auto& tensor : tensors_) {
            if (tensor.name == name) return &tensor;
        }
        return nullptr;
    }

    Config config_;
    void* plugin_handle_ = nullptr;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::vector<TensorBuffer> tensors_;
    float* filtered_disparity_ = nullptr;
    float* full_disparity_ = nullptr;
    float* depth_m_ = nullptr;
    float4* cloud_ = nullptr;
    std::size_t full_resolution_capacity_ = 0;
    int full_width_ = 0;
    int full_height_ = 0;
    StereoCalibration last_calibration_{};
    cudaStream_t stream_ = nullptr;
    cudaEvent_t completion_event_ = nullptr;
    cudaEvent_t timing_start_event_ = nullptr;
    cudaEvent_t preprocess_done_event_ = nullptr;
    cudaEvent_t inference_done_event_ = nullptr;
    cudaEvent_t postprocess_done_event_ = nullptr;
    int output_width_ = 0;
    int output_height_ = 0;
    bool initialized_ = false;
};

EssDepthEngine::EssDepthEngine(Config config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

EssDepthEngine::~EssDepthEngine() = default;

bool EssDepthEngine::initialize() { return impl_->initialize(); }
bool EssDepthEngine::isInitialized() const { return impl_->initialized_; }
bool EssDepthEngine::infer(
    const void* left_bgra_ptr_dev, std::size_t left_step_bytes,
    const void* right_bgra_ptr_dev, std::size_t right_step_bytes,
    int source_width, int source_height, const StereoCalibration& calibration,
    InferenceOutput& output) {
    return impl_->infer(
        left_bgra_ptr_dev, left_step_bytes, right_bgra_ptr_dev, right_step_bytes,
        source_width, source_height, calibration, output);
}
bool EssDepthEngine::getOutputStats(OutputStats& stats) const {
    return impl_->getOutputStats(stats);
}
bool EssDepthEngine::getTimingStats(TimingStats& stats) const {
    return impl_->getTimingStats(stats);
}
bool EssDepthEngine::waitForCompletion() const { return impl_->waitForCompletion(); }
int EssDepthEngine::inputWidth() const { return impl_->config_.input_width; }
int EssDepthEngine::inputHeight() const { return impl_->config_.input_height; }
int EssDepthEngine::outputWidth() const { return impl_->output_width_; }
int EssDepthEngine::outputHeight() const { return impl_->output_height_; }
cudaStream_t EssDepthEngine::stream() const { return impl_->stream_; }
cudaEvent_t EssDepthEngine::completionEvent() const { return impl_->completion_event_; }

}  // namespace core
}  // namespace wheel_perception

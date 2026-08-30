#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>

#include <cuda_runtime_api.h>

namespace wheel_perception {
namespace core {

class EssDepthEngine {
public:
    struct Config {
        std::string engine_path;
        std::string plugin_path;
        int input_width = 480;
        int input_height = 288;
        double confidence_threshold = 0.5;
        double min_depth_m = 0.3;
        double max_depth_m = 10.0;
    };

    struct StereoCalibration {
        float fx = 0.0F;
        float fy = 0.0F;
        float cx = 0.0F;
        float cy = 0.0F;
        float baseline_m = 0.0F;
    };

    struct InferenceOutput {
        const float* disparity_ptr_dev = nullptr;
        const float* depth_ptr_dev = nullptr;
        const float4* cloud_ptr_dev = nullptr;
        const float* confidence_ptr_dev = nullptr;
        int width = 0;
        int height = 0;
        int confidence_width = 0;
        int confidence_height = 0;
    };

    struct CloudProbe {
        int u = 0;
        int v = 0;
        float4 point{};
        bool valid = false;
    };

    struct OutputStats {
        std::size_t pixel_count = 0;
        std::size_t valid_disparity_count = 0;
        std::size_t valid_depth_count = 0;
        std::size_t valid_cloud_count = 0;
        std::size_t finite_confidence_count = 0;
        std::size_t invalid_cloud_encoding_mismatch_count = 0;
        std::size_t cloud_coordinate_mismatch_count = 0;
        std::size_t cloud_direction_mismatch_count = 0;
        float min_disparity = 0.0F;
        float max_disparity = 0.0F;
        float mean_disparity = 0.0F;
        float min_confidence = 0.0F;
        float max_confidence = 0.0F;
        float min_depth_m = 0.0F;
        float max_depth_m = 0.0F;
        float mean_depth_m = 0.0F;
        float max_cloud_coordinate_error_m = 0.0F;
        std::array<CloudProbe, 5> cloud_probes{};
    };

    struct TimingStats {
        float preprocess_ms = 0.0F;
        float inference_ms = 0.0F;
        float depth_cloud_ms = 0.0F;
        float total_ms = 0.0F;
    };

    explicit EssDepthEngine(Config config);
    ~EssDepthEngine();

    bool initialize();
    bool isInitialized() const;
    bool infer(
        const void* left_bgra_ptr_dev, std::size_t left_step_bytes,
        const void* right_bgra_ptr_dev, std::size_t right_step_bytes,
        int source_width, int source_height, const StereoCalibration& calibration,
        InferenceOutput& output);
    bool getOutputStats(OutputStats& stats) const;
    bool getTimingStats(TimingStats& stats) const;
    bool waitForCompletion() const;
    int inputWidth() const;
    int inputHeight() const;
    int outputWidth() const;
    int outputHeight() const;
    cudaStream_t stream() const;
    cudaEvent_t completionEvent() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace core
}  // namespace wheel_perception

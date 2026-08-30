#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include "wheel_perception/core/zed_driver.hpp"

namespace wheel_perception {
namespace core {

class UsbStereoDriver {
public:
    struct Config {
        std::string device;
        std::string fourcc = "YUYV";
        int capture_width = 3840;
        int capture_height = 1080;
        double fps = 30.0;
        int buffer_size = 1;
        int output_width = 1280;
        int output_height = 720;
        std::string calibration_json;
        bool swap_left_right = true;
        bool require_calibration_size = true;
        double rectify_alpha = 0.0;
        bool auto_reconnect = true;
        int reconnect_interval_ms = 500;
        int reconnect_warmup_frames = 3;

        // Use -1 to leave a V4L2 control at the camera's current value.
        int brightness = -1;
        int contrast = -1;
        int saturation = -1;
        int hue = -1;
        int white_balance_automatic = -1;
        int white_balance_temperature = -1;
        int gamma = -1;
        int gain = -1;
        int power_line_frequency = -1;
        int sharpness = -1;
        int backlight_compensation = -1;
    };

    struct CameraParams {
        float fx;
        float fy;
        float cx;
        float cy;
        float baseline;
    };

    UsbStereoDriver();
    ~UsbStereoDriver();

    UsbStereoDriver(const UsbStereoDriver&) = delete;
    UsbStereoDriver& operator=(const UsbStereoDriver&) = delete;

    bool open(const Config& config);
    void close();
    bool grab(ZedGpuFrame& out_frame);
    bool isOpened() const;
    bool isSdkDepthEnabled() const;
    std::string lastError() const;
    CameraParams getCameraParams() const;
    void retrieveImage(cv::Mat& out_img) const;
    // 一次取出左右目校正图（CPU BGR，output_width x output_height），
    // 用于远程推理传输（双机模式下客户端把图发给服务端）
    void retrieveStereoPair(cv::Mat& out_left_bgr, cv::Mat& out_right_bgr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace core
}  // namespace wheel_perception

#include "wheel_perception/core/usb_stereo_driver.hpp"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>
#include <linux/videodev2.h>
#include <jsoncpp/json/json.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <sys/ioctl.h>
#include <unistd.h>

namespace wheel_perception {
namespace core {
namespace {

constexpr bool shouldSetManualWhiteBalanceTemperature(int automatic_white_balance) {
    return automatic_white_balance == 0;
}

constexpr bool shouldApplyV4l2Control(int value) {
    return value != -1;
}

static_assert(shouldSetManualWhiteBalanceTemperature(0));
static_assert(!shouldSetManualWhiteBalanceTemperature(1));
static_assert(!shouldApplyV4l2Control(-1));
static_assert(shouldApplyV4l2Control(0));

cv::Mat readMatrix(const Json::Value& data, const char* key, int rows, int cols) {
    if (!data.isMember(key)) {
        throw std::runtime_error(std::string("missing calibration key: ") + key);
    }
    const auto& values = data[key];
    if (!values.isArray() || static_cast<int>(values.size()) != rows) {
        throw std::runtime_error(std::string("invalid calibration matrix: ") + key);
    }

    cv::Mat matrix(rows, cols, CV_64F);
    for (int row = 0; row < rows; ++row) {
        if (!values[row].isArray() ||
            static_cast<int>(values[row].size()) != cols) {
            throw std::runtime_error(std::string("invalid calibration matrix: ") + key);
        }
        for (int col = 0; col < cols; ++col) {
            matrix.at<double>(row, col) = values[row][col].asDouble();
        }
    }
    return matrix;
}

cv::Mat readDistortion(const Json::Value& data, const char* key) {
    if (!data.isMember(key) || !data[key].isArray() ||
        data[key].empty() || !data[key][0].isArray() ||
        data[key][0].empty()) {
        throw std::runtime_error(std::string("invalid calibration distortion: ") + key);
    }
    const auto& row = data[key][0];
    cv::Mat distortion(1, static_cast<int>(row.size()), CV_64F);
    for (int col = 0; col < distortion.cols; ++col) {
        distortion.at<double>(0, col) = row[col].asDouble();
    }
    return distortion;
}

cv::Mat scaleCameraMatrix(
    const cv::Mat& matrix, const cv::Size& from, const cv::Size& to) {
    cv::Mat scaled = matrix.clone();
    const double scale_x =
        static_cast<double>(to.width) / static_cast<double>(from.width);
    const double scale_y =
        static_cast<double>(to.height) / static_cast<double>(from.height);
    scaled.at<double>(0, 0) *= scale_x;
    scaled.at<double>(0, 2) *= scale_x;
    scaled.at<double>(1, 1) *= scale_y;
    scaled.at<double>(1, 2) *= scale_y;
    return scaled;
}

cv::Mat scaleProjectionMatrix(
    const cv::Mat& matrix, const cv::Size& from, const cv::Size& to) {
    cv::Mat scaled = matrix.clone();
    const double scale_x =
        static_cast<double>(to.width) / static_cast<double>(from.width);
    const double scale_y =
        static_cast<double>(to.height) / static_cast<double>(from.height);
    for (int col = 0; col < scaled.cols; ++col) {
        scaled.at<double>(0, col) *= scale_x;
        scaled.at<double>(1, col) *= scale_y;
    }
    return scaled;
}

std::string fourccString(double value) {
    const int code = static_cast<int>(std::llround(value));
    std::string result(4, '\0');
    result[0] = static_cast<char>(code & 0xff);
    result[1] = static_cast<char>((code >> 8) & 0xff);
    result[2] = static_cast<char>((code >> 16) & 0xff);
    result[3] = static_cast<char>((code >> 24) & 0xff);
    return result;
}

bool setV4l2Control(
    int fd, std::uint32_t id, int value, const char* name, std::string& error) {
    // A value of -1 means "leave the camera's current setting unchanged".
    // All USB control parameters use this sentinel as their default.
    if (!shouldApplyV4l2Control(value)) {
        return true;
    }

    v4l2_queryctrl query{};
    query.id = id;
    if (::ioctl(fd, VIDIOC_QUERYCTRL, &query) != 0) {
        if (errno == EINVAL) {
            std::cerr << "[UsbStereoDriver] V4L2 control unavailable: " << name
                      << std::endl;
            return true;
        }
        error = std::string("VIDIOC_QUERYCTRL failed for ") + name + ": " +
            std::strerror(errno);
        return false;
    }
    if ((query.flags & V4L2_CTRL_FLAG_DISABLED) != 0U) {
        std::cerr << "[UsbStereoDriver] V4L2 control disabled: " << name << std::endl;
        return true;
    }
    if (value < query.minimum || value > query.maximum) {
        error = std::string("V4L2 control value out of range for ") + name +
            ": requested=" + std::to_string(value) +
            " supported=[" + std::to_string(query.minimum) + ", " +
            std::to_string(query.maximum) + "]";
        return false;
    }

    v4l2_control control{};
    control.id = id;
    control.value = value;
    if (::ioctl(fd, VIDIOC_S_CTRL, &control) != 0) {
        error = std::string("VIDIOC_S_CTRL failed for ") + name + ": " +
            std::strerror(errno);
        return false;
    }
    return true;
}

bool applyV4l2Controls(
    const UsbStereoDriver::Config& config, std::string& error) {
    const int fd = ::open(config.device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) {
        error = "failed to open V4L2 controls on " + config.device + ": " +
            std::strerror(errno);
        return false;
    }

    const std::pair<std::uint32_t, std::pair<int, const char*>> controls[] = {
        {V4L2_CID_BRIGHTNESS, {config.brightness, "brightness"}},
        {V4L2_CID_CONTRAST, {config.contrast, "contrast"}},
        {V4L2_CID_SATURATION, {config.saturation, "saturation"}},
        {V4L2_CID_HUE, {config.hue, "hue"}},
        {V4L2_CID_AUTO_WHITE_BALANCE,
         {config.white_balance_automatic, "white_balance_automatic"}},
        {V4L2_CID_GAMMA, {config.gamma, "gamma"}},
        {V4L2_CID_GAIN, {config.gain, "gain"}},
        {V4L2_CID_POWER_LINE_FREQUENCY,
         {config.power_line_frequency, "power_line_frequency"}},
        {V4L2_CID_SHARPNESS, {config.sharpness, "sharpness"}},
        {V4L2_CID_BACKLIGHT_COMPENSATION,
         {config.backlight_compensation, "backlight_compensation"}},
    };

    bool ok = true;
    for (const auto& item : controls) {
        if (!setV4l2Control(
                fd, item.first, item.second.first, item.second.second, error)) {
            ok = false;
            break;
        }

        if (item.first == V4L2_CID_AUTO_WHITE_BALANCE &&
            shouldApplyV4l2Control(config.white_balance_automatic)) {
            if (shouldSetManualWhiteBalanceTemperature(
                    config.white_balance_automatic)) {
                if (!setV4l2Control(
                        fd, V4L2_CID_WHITE_BALANCE_TEMPERATURE,
                        config.white_balance_temperature,
                        "white_balance_temperature", error)) {
                    ok = false;
                    break;
                }
            } else {
                std::cout
                    << "[UsbStereoDriver] Automatic white balance enabled; "
                       "skipping manual white_balance_temperature"
                    << std::endl;
            }
        }
    }
    ::close(fd);
    return ok;
}

std::uint64_t systemTimestampNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

}  // namespace

struct UsbStereoDriver::Impl {
    Config config;
    cv::VideoCapture capture;

    cv::Size calibration_size;
    cv::Mat k_left;
    cv::Mat d_left;
    cv::Mat k_right;
    cv::Mat d_right;
    cv::Mat rotation;
    cv::Mat translation;
    cv::Mat left_map_x;
    cv::Mat left_map_y;
    cv::Mat right_map_x;
    cv::Mat right_map_y;
    cv::Mat p1_output;
    cv::Mat p2_output;

    cv::Mat side_by_side_bgr;
    cv::Mat left_rect_bgr;
    cv::Mat right_rect_bgr;
    cv::Mat left_rect_bgra;
    cv::Mat right_rect_bgra;

    void* left_dev = nullptr;
    void* right_dev = nullptr;
    std::size_t left_pitch = 0;
    std::size_t right_pitch = 0;
    cudaStream_t upload_stream = nullptr;

    CameraParams camera_params{0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    std::string last_error;
    bool logged_frame = false;
    bool initialized = false;
    bool reconnect_pending = false;
    std::uint64_t reconnect_attempts = 0U;
    std::chrono::steady_clock::time_point next_reconnect_attempt{};

    bool loadCalibration() {
        std::ifstream stream(config.calibration_json);
        if (!stream) {
            last_error = "failed to open calibration JSON: " + config.calibration_json;
            return false;
        }

        try {
            Json::Value data;
            Json::CharReaderBuilder builder;
            std::string parse_errors;
            if (!Json::parseFromStream(builder, stream, &data, &parse_errors)) {
                throw std::runtime_error(parse_errors);
            }
            if (!data.isMember("image_width") || !data.isMember("image_height")) {
                throw std::runtime_error("missing calibration image dimensions");
            }
            if (data.isMember("swap_left_right") &&
                data["swap_left_right"].asBool() != config.swap_left_right) {
                throw std::runtime_error(
                    "swap_left_right does not match calibration JSON");
            }
            calibration_size = cv::Size(
                data["image_width"].asInt(),
                data["image_height"].asInt());
            k_left = readMatrix(data, "K_left", 3, 3);
            d_left = readDistortion(data, "D_left");
            k_right = readMatrix(data, "K_right", 3, 3);
            d_right = readDistortion(data, "D_right");
            rotation = readMatrix(data, "R", 3, 3);
            translation = readMatrix(data, "T", 3, 1);
        } catch (const std::exception& error) {
            last_error = std::string("invalid calibration JSON: ") + error.what();
            return false;
        }
        return true;
    }

    bool buildRectificationMaps() {
        const cv::Size source_size(config.capture_width / 2, config.capture_height);
        if (config.require_calibration_size && source_size != calibration_size) {
            last_error =
                "per-eye capture size " + std::to_string(source_size.width) + "x" +
                std::to_string(source_size.height) +
                " does not match calibration size " +
                std::to_string(calibration_size.width) + "x" +
                std::to_string(calibration_size.height);
            return false;
        }

        try {
            const cv::Mat scaled_k_left =
                scaleCameraMatrix(k_left, calibration_size, source_size);
            const cv::Mat scaled_k_right =
                scaleCameraMatrix(k_right, calibration_size, source_size);
            cv::Mat r1;
            cv::Mat r2;
            cv::Mat p1;
            cv::Mat p2;
            cv::Mat q;
            cv::Rect roi1;
            cv::Rect roi2;
            cv::stereoRectify(
                scaled_k_left, d_left, scaled_k_right, d_right, source_size,
                rotation, translation, r1, r2, p1, p2, q,
                cv::CALIB_ZERO_DISPARITY, config.rectify_alpha, source_size,
                &roi1, &roi2);

            const cv::Size output_size(config.output_width, config.output_height);
            p1_output = scaleProjectionMatrix(p1, source_size, output_size);
            p2_output = scaleProjectionMatrix(p2, source_size, output_size);
            const cv::Mat p1_camera =
                p1_output(cv::Rect(0, 0, 3, 3)).clone();
            const cv::Mat p2_camera =
                p2_output(cv::Rect(0, 0, 3, 3)).clone();
            cv::initUndistortRectifyMap(
                scaled_k_left, d_left, r1, p1_camera, output_size, CV_32FC1,
                left_map_x, left_map_y);
            cv::initUndistortRectifyMap(
                scaled_k_right, d_right, r2, p2_camera, output_size, CV_32FC1,
                right_map_x, right_map_y);
        } catch (const cv::Exception& error) {
            last_error = std::string("failed to build rectification maps: ") + error.what();
            return false;
        }

        const double fx = p1_output.at<double>(0, 0);
        const double fy = p1_output.at<double>(1, 1);
        const double cx = p1_output.at<double>(0, 2);
        const double cy = p1_output.at<double>(1, 2);
        const double right_tx = p2_output.at<double>(0, 3);
        const double baseline = -right_tx / p2_output.at<double>(0, 0);
        if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) ||
            !std::isfinite(cy) || !std::isfinite(baseline) ||
            fx <= 0.0 || fy <= 0.0 || baseline <= 0.0) {
            last_error = "rectified calibration contains invalid values";
            return false;
        }
        camera_params = {
            static_cast<float>(fx), static_cast<float>(fy),
            static_cast<float>(cx), static_cast<float>(cy),
            static_cast<float>(baseline)};
        return true;
    }

    bool allocateCudaBuffers() {
        cudaError_t error = cudaStreamCreateWithFlags(
            &upload_stream, cudaStreamNonBlocking);
        if (error != cudaSuccess) {
            last_error = std::string("cudaStreamCreateWithFlags failed: ") +
                cudaGetErrorString(error);
            return false;
        }
        const std::size_t row_bytes =
            static_cast<std::size_t>(config.output_width) * sizeof(uchar4);
        error = cudaMallocPitch(
            &left_dev, &left_pitch, row_bytes,
            static_cast<std::size_t>(config.output_height));
        if (error == cudaSuccess) {
            error = cudaMallocPitch(
                &right_dev, &right_pitch, row_bytes,
                static_cast<std::size_t>(config.output_height));
        }
        if (error != cudaSuccess) {
            last_error = std::string("cudaMallocPitch failed: ") +
                cudaGetErrorString(error);
            return false;
        }
        return true;
    }

    void releaseCudaBuffers() {
        if (upload_stream != nullptr) {
            cudaStreamSynchronize(upload_stream);
        }
        if (left_dev != nullptr) {
            cudaFree(left_dev);
            left_dev = nullptr;
        }
        if (right_dev != nullptr) {
            cudaFree(right_dev);
            right_dev = nullptr;
        }
        left_pitch = 0;
        right_pitch = 0;
        if (upload_stream != nullptr) {
            cudaStreamDestroy(upload_stream);
            upload_stream = nullptr;
        }
    }

    void releaseCapture() {
        if (capture.isOpened()) {
            capture.release();
        }
    }

    bool validateDecodedFrame(const cv::Mat& frame, const char* context) {
        if (!frame.empty() &&
            frame.cols == config.capture_width &&
            frame.rows == config.capture_height &&
            frame.type() == CV_8UC3) {
            return true;
        }
        last_error =
            std::string(context) + ": size=" + std::to_string(frame.cols) + "x" +
            std::to_string(frame.rows) + " type=" + std::to_string(frame.type());
        return false;
    }

    bool openCapture(bool warmup) {
        releaseCapture();
        last_error.clear();
        if (!applyV4l2Controls(config, last_error)) {
            return false;
        }
        if (!capture.open(config.device, cv::CAP_V4L2) || !capture.isOpened()) {
            last_error = "failed to open camera: " + config.device;
            releaseCapture();
            return false;
        }

        capture.set(
            cv::CAP_PROP_FOURCC,
            cv::VideoWriter::fourcc(
                config.fourcc[0], config.fourcc[1],
                config.fourcc[2], config.fourcc[3]));
        capture.set(cv::CAP_PROP_FRAME_WIDTH, config.capture_width);
        capture.set(cv::CAP_PROP_FRAME_HEIGHT, config.capture_height);
        capture.set(cv::CAP_PROP_FPS, config.fps);
        capture.set(cv::CAP_PROP_BUFFERSIZE, config.buffer_size);

        const int actual_width =
            static_cast<int>(std::llround(capture.get(cv::CAP_PROP_FRAME_WIDTH)));
        const int actual_height =
            static_cast<int>(std::llround(capture.get(cv::CAP_PROP_FRAME_HEIGHT)));
        const double actual_fps = capture.get(cv::CAP_PROP_FPS);
        const std::string actual_fourcc =
            fourccString(capture.get(cv::CAP_PROP_FOURCC));
        if (actual_width != config.capture_width ||
            actual_height != config.capture_height ||
            std::abs(actual_fps - config.fps) > 0.5 ||
            actual_fourcc != config.fourcc) {
            last_error =
                "camera mode mismatch: requested=" +
                std::to_string(config.capture_width) + "x" +
                std::to_string(config.capture_height) + "@" +
                std::to_string(config.fps) + " " + config.fourcc +
                " actual=" + std::to_string(actual_width) + "x" +
                std::to_string(actual_height) + "@" +
                std::to_string(actual_fps) + " " + actual_fourcc;
            releaseCapture();
            return false;
        }

        if (warmup) {
            for (int index = 0; index < config.reconnect_warmup_frames; ++index) {
                cv::Mat frame;
                if (!capture.read(frame) ||
                    !validateDecodedFrame(frame, "invalid reconnect warmup frame")) {
                    if (last_error.empty()) {
                        last_error = "failed to read reconnect warmup frame";
                    }
                    releaseCapture();
                    return false;
                }
            }
        }

        std::cout
            << "[UsbStereoDriver] Opened " << config.device << " as "
            << actual_width << "x" << actual_height << "@" << actual_fps << " "
            << actual_fourcc << "; rectified=" << config.output_width << "x"
            << config.output_height << " BGRA swap_left_right="
            << (config.swap_left_right ? "true" : "false") << std::endl;
        return true;
    }

    void scheduleReconnect(const std::string& reason) {
        releaseCapture();
        reconnect_pending = config.auto_reconnect;
        next_reconnect_attempt =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(config.reconnect_interval_ms);
        last_error = reason;
        if (config.auto_reconnect) {
            last_error +=
                "; reconnect scheduled in " +
                std::to_string(config.reconnect_interval_ms) + " ms";
        }
    }

    bool reconnectIfDue() {
        if (capture.isOpened()) {
            return true;
        }
        if (!initialized) {
            last_error = "USB stereo driver is not initialized";
            return false;
        }
        if (!config.auto_reconnect || !reconnect_pending) {
            last_error = "camera is disconnected and automatic reconnect is disabled";
            return false;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now < next_reconnect_attempt) {
            return false;
        }

        ++reconnect_attempts;
        if (!openCapture(true)) {
            const std::string attempt_error = last_error;
            next_reconnect_attempt =
                std::chrono::steady_clock::now() +
                std::chrono::milliseconds(config.reconnect_interval_ms);
            last_error =
                "USB reconnect attempt " + std::to_string(reconnect_attempts) +
                " failed: " + attempt_error;
            std::cerr << "[UsbStereoDriver] " << last_error << std::endl;
            return false;
        }

        std::cout
            << "[UsbStereoDriver] Camera reconnected after "
            << reconnect_attempts << " attempt(s); discarded "
            << config.reconnect_warmup_frames << " warmup frame(s)" << std::endl;
        reconnect_pending = false;
        reconnect_attempts = 0U;
        logged_frame = false;
        last_error.clear();
        return true;
    }
};

UsbStereoDriver::UsbStereoDriver() : pimpl_(std::make_unique<Impl>()) {}

UsbStereoDriver::~UsbStereoDriver() {
    close();
}

bool UsbStereoDriver::open(const Config& config) {
    if (isOpened()) {
        return true;
    }
    close();
    pimpl_->config = config;
    pimpl_->last_error.clear();
    pimpl_->logged_frame = false;
    pimpl_->initialized = false;
    pimpl_->reconnect_pending = false;
    pimpl_->reconnect_attempts = 0U;

    if (config.device.empty() || config.fourcc.size() != 4 ||
        config.capture_width <= 0 || config.capture_height <= 0 ||
        config.capture_width % 2 != 0 || config.fps <= 0.0 ||
        config.buffer_size <= 0 || config.output_width <= 0 ||
        config.output_height <= 0 || config.calibration_json.empty() ||
        config.reconnect_interval_ms <= 0 ||
        config.reconnect_warmup_frames < 0) {
        pimpl_->last_error = "invalid USB stereo driver configuration";
        return false;
    }
    if (!pimpl_->loadCalibration() || !pimpl_->buildRectificationMaps()) {
        return false;
    }
    if (!pimpl_->openCapture(false)) {
        return false;
    }

    if (!pimpl_->allocateCudaBuffers()) {
        close();
        return false;
    }

    std::cout
        << "[UsbStereoDriver] Calibration: fx=" << pimpl_->camera_params.fx
        << " fy=" << pimpl_->camera_params.fy
        << " cx=" << pimpl_->camera_params.cx
        << " cy=" << pimpl_->camera_params.cy
        << " baseline=" << pimpl_->camera_params.baseline << " m" << std::endl;
    pimpl_->initialized = true;
    pimpl_->last_error.clear();
    return true;
}

void UsbStereoDriver::close() {
    if (!pimpl_) {
        return;
    }
    pimpl_->releaseCapture();
    pimpl_->releaseCudaBuffers();
    pimpl_->side_by_side_bgr.release();
    pimpl_->left_rect_bgr.release();
    pimpl_->right_rect_bgr.release();
    pimpl_->left_rect_bgra.release();
    pimpl_->right_rect_bgra.release();
    pimpl_->initialized = false;
    pimpl_->reconnect_pending = false;
    pimpl_->reconnect_attempts = 0U;
}

bool UsbStereoDriver::grab(ZedGpuFrame& out_frame) {
    out_frame = ZedGpuFrame{};
    if (!pimpl_->reconnectIfDue()) {
        return false;
    }

    cv::Mat frame;
    if (!pimpl_->capture.read(frame) || frame.empty()) {
        pimpl_->scheduleReconnect("failed to read USB stereo frame");
        return false;
    }
    if (!pimpl_->validateDecodedFrame(frame, "unexpected decoded frame")) {
        const std::string validation_error = pimpl_->last_error;
        pimpl_->scheduleReconnect(validation_error);
        return false;
    }

    const int half_width = frame.cols / 2;
    const cv::Mat video_left =
        frame(cv::Rect(0, 0, half_width, frame.rows));
    const cv::Mat video_right =
        frame(cv::Rect(half_width, 0, half_width, frame.rows));
    const cv::Mat& logical_left =
        pimpl_->config.swap_left_right ? video_right : video_left;
    const cv::Mat& logical_right =
        pimpl_->config.swap_left_right ? video_left : video_right;

    try {
        cv::remap(
            logical_left, pimpl_->left_rect_bgr, pimpl_->left_map_x,
            pimpl_->left_map_y, cv::INTER_LINEAR);
        cv::remap(
            logical_right, pimpl_->right_rect_bgr, pimpl_->right_map_x,
            pimpl_->right_map_y, cv::INTER_LINEAR);
        cv::cvtColor(
            pimpl_->left_rect_bgr, pimpl_->left_rect_bgra, cv::COLOR_BGR2BGRA);
        cv::cvtColor(
            pimpl_->right_rect_bgr, pimpl_->right_rect_bgra, cv::COLOR_BGR2BGRA);
    } catch (const cv::Exception& error) {
        pimpl_->last_error =
            std::string("USB stereo rectification failed: ") + error.what();
        return false;
    }

    const std::size_t row_bytes =
        static_cast<std::size_t>(pimpl_->config.output_width) * sizeof(uchar4);
    cudaError_t error = cudaMemcpy2DAsync(
        pimpl_->left_dev, pimpl_->left_pitch,
        pimpl_->left_rect_bgra.data, pimpl_->left_rect_bgra.step,
        row_bytes, static_cast<std::size_t>(pimpl_->config.output_height),
        cudaMemcpyHostToDevice, pimpl_->upload_stream);
    if (error == cudaSuccess) {
        error = cudaMemcpy2DAsync(
            pimpl_->right_dev, pimpl_->right_pitch,
            pimpl_->right_rect_bgra.data, pimpl_->right_rect_bgra.step,
            row_bytes, static_cast<std::size_t>(pimpl_->config.output_height),
            cudaMemcpyHostToDevice, pimpl_->upload_stream);
    }
    if (error == cudaSuccess) {
        error = cudaStreamSynchronize(pimpl_->upload_stream);
    }
    if (error != cudaSuccess) {
        pimpl_->last_error =
            std::string("USB stereo CUDA upload failed: ") +
            cudaGetErrorString(error);
        return false;
    }

    out_frame.timestamp_ns = systemTimestampNs();
    out_frame.rgb_ptr_dev = pimpl_->left_dev;
    out_frame.rgb_step_bytes = static_cast<int>(pimpl_->left_pitch);
    out_frame.right_ptr_dev = pimpl_->right_dev;
    out_frame.right_step_bytes = static_cast<int>(pimpl_->right_pitch);
    out_frame.width = pimpl_->config.output_width;
    out_frame.height = pimpl_->config.output_height;
    out_frame.pose_valid = false;
    out_frame.imu_valid = false;

    if (!pimpl_->logged_frame) {
        std::cout
            << "[UsbStereoDriver] Stereo GPU frame ready: " << out_frame.width
            << "x" << out_frame.height
            << " left_step=" << out_frame.rgb_step_bytes
            << " right_step=" << out_frame.right_step_bytes
            << " timestamp_ns=" << out_frame.timestamp_ns << std::endl;
        pimpl_->logged_frame = true;
    }
    pimpl_->last_error.clear();
    return true;
}

bool UsbStereoDriver::isOpened() const {
    return pimpl_ && pimpl_->capture.isOpened();
}

bool UsbStereoDriver::isSdkDepthEnabled() const {
    return false;
}

std::string UsbStereoDriver::lastError() const {
    return pimpl_ ? pimpl_->last_error : "driver is not initialized";
}

UsbStereoDriver::CameraParams UsbStereoDriver::getCameraParams() const {
    return pimpl_ ? pimpl_->camera_params : CameraParams{0, 0, 0, 0, 0};
}

void UsbStereoDriver::retrieveImage(cv::Mat& out_img) const {
    if (!pimpl_ || pimpl_->left_rect_bgr.empty()) {
        out_img.release();
        return;
    }
    pimpl_->left_rect_bgr.copyTo(out_img);
}

void UsbStereoDriver::retrieveStereoPair(cv::Mat& out_left_bgr,
                                         cv::Mat& out_right_bgr) const {
    if (!pimpl_ || pimpl_->left_rect_bgr.empty() || pimpl_->right_rect_bgr.empty()) {
        out_left_bgr.release();
        out_right_bgr.release();
        return;
    }
    pimpl_->left_rect_bgr.copyTo(out_left_bgr);
    pimpl_->right_rect_bgr.copyTo(out_right_bgr);
}

}  // namespace core
}  // namespace wheel_perception

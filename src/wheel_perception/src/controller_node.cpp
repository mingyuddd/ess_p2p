#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include "wheel_msgs/msg/perception_output.hpp"
#include "wheel_perception/core/lqr_controller.hpp"

#include "wheel_perception/core/zed_driver.hpp"



namespace wheel_control {

enum class State {
    STOP, CRUISE, AVOID_HARD_LEFT, AVOID_HARD_RIGHT, AVOID_SLOW, CONTINUE_STRAIGHT, RECOVER_TURN, OUT
};

class ControllerNode : public rclcpp::Node {
public:
    explicit ControllerNode(const rclcpp::NodeOptions & options)
        : Node("controller_node", options) 
    {
        // === 参数配置 (严格对齐 avoid_lib_s.cpp) ===
        this->declare_parameter("lqr.gain", 60.0);    // 协议放大倍数
        this->declare_parameter("lqr.q_pos", 10.0);   // Q(0,0)
        this->declare_parameter("lqr.q_ang", 10.0);   // Q(2,2)
        this->declare_parameter("lqr.q_integral", 0.0); // Q(4,4)
        this->declare_parameter("lqr.integral_limit", 1.5);
        this->declare_parameter("lqr.k_w", 10.0);     // k_w
        this->declare_parameter("lqr.model_v", 0.5);  // [关键] 模型内部速度 static v=0.5
        this->declare_parameter("lqr.aim_dist", 0.65);
        this->declare_parameter("dynamic_aim.enabled", true);

        this->declare_parameter("logic.base_vel", 1.0); // 实际行驶速度 linear.x=1.0
        this->declare_parameter("logic.stop_dist", 1.0);
        this->declare_parameter("logic.narrow_road_width", 3.0);
        this->declare_parameter("logic.pass_clearance", 0.6);
        this->declare_parameter("logic.recover_time", 2.0);
        this->declare_parameter("logic.straight_speed_est", 0.8); // 直行模式估算车速 (m/s)：里程计不可用时按 车速×时间 估算通过距离

        // 初始化 LQR
        LqrController::Config lqr_cfg;
        lqr_cfg.q_pos = this->get_parameter("lqr.q_pos").as_double();
        lqr_cfg.q_ang = this->get_parameter("lqr.q_ang").as_double();
        lqr_cfg.q_vel = 0.0; // dot_e (旧代码默认0)
        lqr_cfg.q_ang_vel = 0.0; // dot_th_e (旧代码默认0)
        lqr_cfg.q_integral = this->get_parameter("lqr.q_integral").as_double();
        lqr_cfg.lqr_gain = this->get_parameter("lqr.gain").as_double();
        lqr_cfg.k_w = this->get_parameter("lqr.k_w").as_double();
        lqr_cfg.model_v = this->get_parameter("lqr.model_v").as_double(); // 0.5
        lqr_cfg.integral_limit = this->get_parameter("lqr.integral_limit").as_double();
        lqr_cfg.dt = 0.1;
        lqr_ = std::make_unique<LqrController>(lqr_cfg);

        // 通信
        sub_perception_ = this->create_subscription<wheel_msgs::msg::PerceptionOutput>(
            "perception/output", 10, std::bind(&ControllerNode::perception_callback, this, std::placeholders::_1));
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&ControllerNode::odom_callback, this, std::placeholders::_1));
        pub_cmd_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
        
        RCLCPP_INFO(get_logger(), "Controller Aligned: Model_V=%.1f, Real_V=%.1f", lqr_cfg.model_v, this->get_parameter("logic.base_vel").as_double());
    }

private:
    std::unique_ptr<LqrController> lqr_;
    rclcpp::Subscription<wheel_msgs::msg::PerceptionOutput>::SharedPtr sub_perception_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_;
    State state_ = State::CRUISE;
    const char * state_name(State s) const {
        switch (s) {
            case State::STOP: return "STOP";
            case State::CRUISE: return "CRUISE";
            case State::AVOID_HARD_LEFT: return "AVOID_HARD_LEFT";
            case State::AVOID_HARD_RIGHT: return "AVOID_HARD_RIGHT";
            case State::AVOID_SLOW: return "AVOID_SLOW";
            case State::CONTINUE_STRAIGHT: return "CONTINUE_STRAIGHT";
            case State::RECOVER_TURN: return "RECOVER_TURN";
            case State::OUT: return "OUT";
            default: return "UNKNOWN";
        }
    }

    double obs_global_x_ = 0.0;
    bool has_locked_obs_ = false;
    bool narrow_road_stop_ = false;
    rclcpp::Time recover_start_time_;
    rclcpp::Time straight_start_time_;

    //这个回调是实时模式
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_x_ = msg->pose.pose.position.x;
        current_y_ = msg->pose.pose.position.y;  // 往右边走是负数
        current_z_ = msg->pose.pose.position.z;

        // 四元数 → 欧拉角（RPY）
        const auto & q = msg->pose.pose.orientation;
        tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
        tf2::Matrix3x3(tf_q).getRPY(current_roll_, current_pitch_, current_yaw_);
        current_roll_ = current_roll_ * 180.0 / M_PI;
        current_pitch_ = current_pitch_ * 180.0 / M_PI;
        current_yaw_ = current_yaw_ * 180.0 / M_PI;
    }



void perception_callback(const wheel_msgs::msg::PerceptionOutput::SharedPtr msg) {
        geometry_msgs::msg::Twist cmd;
        
        // 动态读取参数
        double base_vel = this->get_parameter("logic.base_vel").as_double();  //·行驶速度1
        double aim_dist = this->get_parameter("lqr.aim_dist").as_double();  //保持跟右侧的距离1
        double stop_dist = this->get_parameter("logic.stop_dist").as_double();  //停车距离1
        double narrow_road_width = this->get_parameter("logic.narrow_road_width").as_double();
        if(!use_dataset_mode_){
            x_min = msg->min_point.x; 
            y_min = msg->min_point.y; 
            min_d = msg->min_distance;
            // RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200, "MinPoint: (%.2f, %.2f)", current_x_, current_y_);
        }
        // RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200, "State: %s | MinDist: %.2f | MinPoint: (%.2f, %.2f)", state_name(state_), min_d, x_min, y_min);
        // 2. 紧急停车
        const bool obstacle_in_avoid_range = (min_d < 2.0 && min_d > 0.01);
        const bool narrow_road = msg->has_road_width && msg->road_width > 0.01 && msg->road_width < narrow_road_width;
        if(rclcpp::ok() && (out_left0 || out_right0 || out_left1 || out_right1) && (state_ != State::CONTINUE_STRAIGHT)) state_ = State::OUT;
        if(rclcpp::ok() && obstacle_in_avoid_range && narrow_road) {
            narrow_road_stop_ = true;
            state_ = State::STOP;
        }
        if(rclcpp::ok() && min_d < stop_dist && min_d > 0.01 && (y_min < 0.65 && y_min > -0.3)) state_ = State::STOP;
        // RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200,"Current State: %s ", state_name(state_));
        

        switch (state_) {
            case State::STOP:
                {
                x_last = 0.0; y_last = 0.0;
                flag_obstacle = true;
                out_left0 = false; out_right0 = false; out_left1 = false; out_right1 = false;
                cmd.linear.x = 0.0; cmd.angular.z = 0.0;
                const bool was_narrow_road_stop = narrow_road_stop_;
                if ((narrow_road_stop_ && !obstacle_in_avoid_range) ||
                    (!narrow_road_stop_ && min_d > stop_dist)) {
                    narrow_road_stop_ = false;
                    state_ = State::CRUISE; lqr_->reset(); has_locked_obs_ = false; 
                }
                if (was_narrow_road_stop) {
                    RCLCPP_INFO_THROTTLE(
                        get_logger(), *this->get_clock(), 200,
                        "STOP! Narrow road width=%.2f, obstacle=%.2f meters!",
                        msg->road_width, min_d);
                } else {
                    RCLCPP_INFO_THROTTLE(
                        get_logger(), *this->get_clock(), 200,
                        "STOP! Obstacle at %.2f meters!", min_d);
                }
                break;
                }

            case State::OUT:
                flag_obstacle = false;
                cmd.linear.x = 0.5;
                if(out_left0) {
                cmd.angular.z = -100.0;
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 200, 
                        "==> 建议：向左转弯");
                }
                else if(out_right0) {
                cmd.angular.z = 100.0;
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 200, 
                        "==> 建议：向右转弯");
                }
                else if(out_left1) {
                    cmd.linear.x = 0.0;
                    cmd.angular.z = -1000.0;
                } 
                else if(out_right1) {
                    cmd.linear.x = 0.0;
                    cmd.angular.z = 1000.0;
                }
                state_ = State::CRUISE;
                break;

            case State::CRUISE: //巡线+判断避障
                if (obstacle_in_avoid_range) {
                    flag_obstacle = true;
                    out_left0 = false; out_right0 = false; out_left1 = false; out_right1 = false;
                    if (y_min < 0.5 && y_min > -0.3) { 
                        state_ = State::AVOID_HARD_LEFT;
                    } 
                    else if ((y_min >= 0.5) && (y_min < 0.8)) { 
                        state_ = State::AVOID_HARD_RIGHT;
                    }
                    else if ((y_min >= 0.8) && (y_min < 1.5)) { 
                        state_ = State::AVOID_SLOW;
                    }
                } else {
                    flag_obstacle = false;
                    // --- LQR 巡线 ---
                    if (msg->has_road_edge) {
                        // 1. 角度单位转换
                        double yaw_err_rad = msg->road_yaw_error * M_PI / 180.0;
                        
                        // 动态目标有效时优先使用；关闭功能或目标无效时回退固定 aim_dist。
                        bool dynamic_aim_enabled = this->get_parameter("dynamic_aim.enabled").as_bool();
                        double target_dist = (dynamic_aim_enabled && msg->target_right_distance > 0.01)
                            ? msg->target_right_distance
                            : aim_dist;
                        double dist_err = msg->right_distance - target_dist;
                        
                        // 3. 计算 (正输入，LQR内部负反馈)
                        cmd.angular.z = lqr_->compute(dist_err, yaw_err_rad); 
                        RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200,
                         "right=%.2f width=%.2f target=%.2f angular=%.2f",
                         msg->right_distance, msg->road_width, target_dist, cmd.angular.z);
                    }
                    cmd.linear.x = base_vel;
                }
                break;

            case State::AVOID_HARD_LEFT:
                {
                   double x_limit = std::min(x_min, 2.0);
                   cmd.angular.z = -70.0 - 20.0 * (2.0 - x_limit);
                   cmd.linear.x = base_vel;
                   if(x_min > 0.01 && x_min < 2.0) {
                        x_last = x_min;
                        y_last = y_min;
                   }
                   RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200, "TURN LEFT");
                   if (min_d == 10.0) {
                        straight_start_time_ = this->now();   // [改] 不依赖里程计：进入直行时记录起点时间
                       state_ = State::CONTINUE_STRAIGHT;
                   }
                }
                break;

            case State::AVOID_HARD_RIGHT:
                {
                   double x_limit = std::min(x_min, 2.0);
                   cmd.angular.z = 50.0;
                   cmd.linear.x = base_vel;
                   if(x_min > 0.01 && x_min < 2.0) {
                        x_last = x_min;
                        y_last = y_min;
                   }
                   if (min_d == 10.0) {
                        straight_start_time_ = this->now();   // [改] 不依赖里程计：进入直行时记录起点时间
                       state_ = State::CONTINUE_STRAIGHT;
                   }
                   RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200, "TURN RIGHT");
                }
                break;
            
            case State::AVOID_SLOW:
                {
                    cmd.linear.x = 0.5; 
                    RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200, "SLOW DOWN");
                    if (msg->has_road_edge) {
                         double yaw_err_rad = msg->road_yaw_error * M_PI / 180.0;
                         // 避障减速状态下仍复用同一动态右侧目标。
                         bool dynamic_aim_enabled = this->get_parameter("dynamic_aim.enabled").as_bool();
                         double target_dist = (dynamic_aim_enabled && msg->target_right_distance > 0.01)
                            ? msg->target_right_distance
                            : aim_dist;
                         double dist_err = msg->right_distance - target_dist;
                         cmd.angular.z = lqr_->compute(dist_err, yaw_err_rad);
                    }
                    if (min_d > 1.5) state_ = State::CRUISE;
                }
                break;

            // ... (RECOVER_STRAIGHT 和 RECOVER_TURN 保持不变) ...
            case State::CONTINUE_STRAIGHT:
                {
                    cmd.linear.x = base_vel;
                    cmd.angular.z = 0.0;
                    // [改] 不依赖里程计：通过距离 = 估算车速 × 直行时长（车速 0.8 m/s，见 logic.straight_speed_est）
                    double speed_est = this->get_parameter("logic.straight_speed_est").as_double();
                    double time_elapsed = (this->now() - straight_start_time_).seconds();
                    double dist_passed = speed_est * time_elapsed;

                    double pass_clearance = this->get_parameter("logic.pass_clearance").as_double();
                    bool odom_ok = (dist_passed > pass_clearance);
                    RCLCPP_INFO_THROTTLE(
                        get_logger(), *this->get_clock(), 200,
                        "CONTINUE STRAIGHT,t=%.2fs,passed=%.2fm,target=%.2fm",
                        time_elapsed, dist_passed, pass_clearance);
                    if (odom_ok || (dist_passed > 3.0)) {
                        state_ = State::CRUISE;
                        x_last = 0.0; y_last = 0.0;
                    }
                }
                break;

            case State::RECOVER_TURN:
                {
                    RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200, "RECOVER TURN");
                    cmd.linear.x = 0.0;
                    cmd.angular.z = 1000.0; 
                    double time_elapsed = (this->now() - recover_start_time_).seconds();
                    if (time_elapsed > this->get_parameter("logic.recover_time").as_double()) {
                        state_ = State::CRUISE;
                        lqr_->reset();
                        RCLCPP_INFO_THROTTLE(get_logger(), *this->get_clock(), 200, "Recover Done.");
                    }
                }
                break;
        }
        pub_cmd_->publish(cmd);

    }
};

} // namespace wheel_control

RCLCPP_COMPONENTS_REGISTER_NODE(wheel_control::ControllerNode)

#ifndef WHEEL_CONTROL_LQR_CONTROLLER_HPP_
#define WHEEL_CONTROL_LQR_CONTROLLER_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace wheel_control {

class LqrController {
public:
    struct Config {
        double q_pos = 10.0;      // Q(0,0) - 位置误差权重
        double q_vel = 0.0;       // Q(1,1) - 位置变化率权重 (对应 dot_e)
        double q_ang = 10.0;      // Q(2,2) - 角度误差权重
        double q_ang_vel = 0.0;   // Q(3,3) - 角度变化率权重 (对应 dot_th_e)
        double q_integral = 0.0;  // Q(4,4) - 位置误差积分权重
        double r_weight = 1.0;    // R(0,0)
        
        double dt = 0.1;          // 采样时间
        double k_w = 10.0;        // B 矩阵参数
        double model_v = 0.5;     // [关键] 模型内部预设速度 (对齐旧代码 static v=0.5)
        double lqr_gain = 60.0;   // 协议增益
        double integral_limit = 1.5; // 位置误差积分限幅，防止 windup
    };

    explicit LqrController(const Config& cfg) : cfg_(cfg) {
        last_dist_ = 0.0;
        last_angle_ = 0.0;
        integral_dist_ = 0.0;
    }

    void reset() {
        last_dist_ = 0.0;
        last_angle_ = 0.0;
        integral_dist_ = 0.0;
    }

    // 计算控制量
    double compute(double dist, double angle) {
        // 1. 预处理 (对齐旧代码 abs判断)
        if (std::abs(dist) < 0.1) dist = 0;
        if (std::abs(angle) < 0.08) angle = 0;

        // 2. 状态向量 x (5x1)
        Eigen::VectorXd x(5);
        double dot_dist = (dist - last_dist_) / cfg_.dt;
        double dot_angle = (angle - last_angle_) / cfg_.dt;
        double integral_limit = std::max(0.0, cfg_.integral_limit);
        integral_dist_ += dist * cfg_.dt;
        integral_dist_ = std::clamp(integral_dist_, -integral_limit, integral_limit);
        
        x << dist, dot_dist, angle, dot_angle, integral_dist_;

        // 3. 模型矩阵 A (5x5)
        // [关键] 使用 cfg_.model_v (0.5) 而不是实际车速 (1.0)
        Eigen::MatrixXd A = Eigen::MatrixXd::Identity(5, 5);
        A(0, 1) = cfg_.dt;
        A(1, 2) = cfg_.model_v;     
        A(2, 3) = cfg_.dt;
        A(4, 0) = cfg_.dt;

        // 4. 输入矩阵 B (5x1)
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(5, 1);
        B(3, 0) = cfg_.k_w;

        // 5. 权重矩阵 Q (5x5)
        Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(5, 5);
        Q(0, 0) = cfg_.q_pos;
        Q(1, 1) = cfg_.q_vel; // default 0.0
        Q(2, 2) = cfg_.q_ang;
        Q(3, 3) = cfg_.q_ang_vel; // default 0.0
        Q(4, 4) = cfg_.q_integral;

        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(1, 1);
        R(0, 0) = cfg_.r_weight;

        // 6. 求解 DARE
        Eigen::MatrixXd P = solve_dare(A, B, Q, R);
        Eigen::MatrixXd K = (B.transpose() * P * B + R).inverse() * (B.transpose() * P * A);

        // 7. 计算输出 u = -Kx
        // 旧代码逻辑: angular_lqr (即 u)
        double u = -(K * x)(0, 0);

        // 更新历史
        last_dist_ = dist;
        last_angle_ = angle;

        // 8. 应用增益
        // 旧代码: -angular_lqr * lqr_k 
        // 也就是: -u * gain = -(-Kx) * gain = Kx * gain
        // 我们的 compute 返回 u (-Kx)。
        // 外部 ControllerNode 会直接使用 compute 的返回值。
        // 如果外部也不加负号，那就是 -Kx * gain (负反馈)。
        // 旧代码 logic: input=-angle -> u_internal=K*angle -> output = -u_internal*k = -K*angle*k (负反馈)
        // 新代码 logic: input=angle  -> u_internal=-K*angle -> output = u_internal*k  = -K*angle*k (负反馈)
        // 结论：完全一致。
        return - ( u * cfg_.lqr_gain ); 
    }

private:
    Config cfg_;
    double last_dist_;
    double last_angle_;
    double integral_dist_;

    Eigen::MatrixXd solve_dare(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B,
                               const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
        Eigen::MatrixXd P = Q;
        Eigen::MatrixXd P_next = Q;
        double eps = 0.01;
        for (int i = 0; i < 100; ++i) {
            Eigen::MatrixXd AtPA = A.transpose() * P * A;
            Eigen::MatrixXd AtPB = A.transpose() * P * B;
            Eigen::MatrixXd BtPA = B.transpose() * P * A;
            Eigen::MatrixXd R_BtPB = R + B.transpose() * P * B;
            P_next = AtPA - AtPB * R_BtPB.inverse() * BtPA + Q;
            if ((P_next - P).cwiseAbs().maxCoeff() < eps) break;
            P = P_next;
        }
        return P_next;
    }
};

} // namespace wheel_control

#endif

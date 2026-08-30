#include "wheel_perception/core/zed_driver.hpp"

bool flag_odom = false,flag_obstacle = false;

// struct MaskRecord {
//     uint8_t mask[ai_w * ai_h];  // BiSeNet argmax 结果（每像素 1 字节类别 ID）
//     float   pose_x, pose_y;     // 记录时刻 ZED 里程计世界坐标 (m)
//     float   yaw;                // 记录时刻航向角 (rad)，由四元数提取
//     float   accum_dist;         // 自上次记录以来的累积位移，用于触发判断
// };

double current_x_ = 0.0,current_y_ = 0.0,current_z_ = 0.0,
       current_roll_  = 0.0,  // 横滚角 (rad)
       current_pitch_ = 0.0,  // 俯仰角 (rad)
       current_yaw_   = 0.0;  // 航向角 (rad)，绕Z轴，左转为正



double x_min,x_last,odx_last = 0.0,ody_last = 0.0; 
double y_min,y_last;
double min_d,right_right;
bool use_dataset_mode_ = true,out_left0 = false,out_right0 = false,out_left1 = false,out_right1 = false;
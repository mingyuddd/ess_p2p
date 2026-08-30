# 立体相机标定参数（校正后内参，与输出分辨率对应）
# 由客户端（采集端）发布，供服务端（远程推理端）使用
std_msgs/Header header
float32 fx
float32 fy
float32 cx
float32 cy
float32 baseline_m

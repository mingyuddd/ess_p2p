# wheel_msgs/msg/PerceptionOutput.msg

std_msgs/Header header

# === 1. 紧急避障 (Stop Logic) ===
# 全局最近障碍物的距离和位置
float32 min_distance
geometry_msgs/Point min_point # (x=前, y=左)

# === 2. 巡线控制 (LQR Logic) ===
# 直接对应 Python 代码中的 rs_msg.right_distance 和 rs_msg.right_angle
bool has_road_edge

# [Lateral Error] 车辆距离右侧路沿的横向距离 (对应 Python 的 right_distance)
# 单位: 米
float32 right_distance

# [Heading Error] 道路走向与车头的夹角 (对应 Python 的 right_angle)
# 建议: 在这里统一转换为弧度 (radians)，Python 代码算的是角度，C++发布时转一下
# 范围: -pi 到 +pi. 0表示平行.
float32 road_yaw_error

# === 2.1 动态右侧目标距离 ===
# 左侧边界只用于估计道路宽度；右侧路沿仍是 LQR 主巡线基准。
bool has_road_width
float32 left_distance
float32 road_width
float32 target_right_distance

# === 3. 调试信息 ===
# 将 Python 中拟合用的两个关键点发出来，方便在 Rviz 画线调试
# 对应 p_mean_1 (远端) 和 p_mean_2 (近端)
geometry_msgs/Point debug_line_pt1
geometry_msgs/Point debug_line_pt2

ObstacleInfo[] obstacles

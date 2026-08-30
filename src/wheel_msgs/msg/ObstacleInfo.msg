# wheel_msgs/msg/ObstacleInfo.msg

int32 class_id           # 0=Road/Bg, 1=Obstacle, etc.
float32 confidence

# 空间位置 (ROS Body Frame: X前, Y左, Z上)
geometry_msgs/Point position

# 3D 尺寸
geometry_msgs/Vector3 size
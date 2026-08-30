import os
import yaml # 引入 yaml 库
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node

def generate_launch_description():
    # 1. 获取 YAML 配置文件路径
    config_path = os.path.join(
        get_package_share_directory('wheel_perception'),
        'config',
        'params.yaml'
    )
    
    # ========================================================================
    # 【核弹操作】直接用 Python 读取 YAML，绕过 ROS 的名字匹配机制
    # ========================================================================
    print(f"\n[Debug] Loading YAML from: {config_path}")
    
    param_dict = {}
    with open(config_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
        
        # 自动探测 YAML 的顶层 Key 是什么
        # 无论是 'fusion_node' 还是 '/fusion_node' 还是 '/**'，我们都把它抓出来
        target_key = None
        if '/**' in yaml_content:
            target_key = '/**'
        elif 'fusion_node' in yaml_content:
            target_key = 'fusion_node'
        elif '/fusion_node' in yaml_content:
            target_key = '/fusion_node'
            
        if target_key:
            # 提取真正的参数字典
            param_dict = yaml_content[target_key]['ros__parameters']
            print(f"[Debug] Successfully loaded {len(param_dict)} parameters under key '{target_key}'")
            # 打印一个参数验证一下
            if 'perception' in param_dict and 'roi' in param_dict['perception']:
                 print(f"[Debug] Loaded min_y: {param_dict['perception']['roi']['min_y']}")
        else:
            print("[Error] Could not find valid key (/**, fusion_node, /fusion_node) in YAML!")

    # ========================================================================

    # 2. 定义 TF 静态变换
    tf_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=['0.0', '0', '0.0', '0', '0', '0', 'base_link', 'zed_left_camera_frame']
    )

    # 3. 定义感知组件 (FusionNode)
    fusion_node_plugin = ComposableNode(
        package='wheel_perception',
        plugin='FusionNode',
        name='fusion_node',
        namespace='',
        # 【关键修改】这里传的是 Python 字典，不是文件路径！
        parameters=[param_dict], 
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # 4. 定义控制组件 (ControllerNode)
    controller_node_plugin = ComposableNode(
        package='wheel_perception',
        plugin='wheel_control::ControllerNode', 
        name='controller_node',
        namespace='',
        # 控制节点也可以用同样的字典，或者继续用文件路径（如果它没问题的话）
        parameters=[param_dict], 
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # 5. 创建容器
    container = ComposableNodeContainer(
        name='wheel_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            fusion_node_plugin,
            controller_node_plugin
        ],
        output='screen',
    )

    # 6. 生命周期管理
    # Composable lifecycle nodes can take a few seconds to become discoverable.
    # Configure is retried first; activate runs only after configure succeeds.
    lifecycle_cmd = ExecuteProcess(
        cmd=[
            'bash',
            '-lc',
            (
                'configured=0; '
                'for i in $(seq 1 30); do '
                '  if ros2 lifecycle set /fusion_node configure; then configured=1; break; fi; '
                '  echo "[lifecycle] waiting for /fusion_node ($i/30)"; '
                '  sleep 1; '
                'done; '
                'if [ "$configured" != "1" ]; then '
                '  echo "[lifecycle] failed to configure /fusion_node"; '
                '  exit 1; '
                'fi; '
                'ros2 lifecycle set /fusion_node activate'
            )
        ],
        output='screen'
    )

    return LaunchDescription([
        tf_publisher,
        container,
        TimerAction(period=2.0, actions=[lifecycle_cmd]),
    ])

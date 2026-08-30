import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def _load_params(config_path):
    with open(config_path, "r") as stream:
        yaml_content = yaml.safe_load(stream)

    for key in ("/**", "fusion_node", "/fusion_node"):
        if key in yaml_content:
            return yaml_content[key]["ros__parameters"]
    raise RuntimeError(f"Could not find ROS parameters in {config_path}")


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory("wheel_perception"),
        "config",
        "params.yaml",
    )
    param_dict = _load_params(config_path)

    tf_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_camera_tf",
        arguments=[
            "0.0",
            "0",
            "0.0",
            "0",
            "0",
            "0",
            "base_link",
            "zed_left_camera_frame",
        ],
    )

    fusion_node_plugin = ComposableNode(
        package="wheel_perception",
        plugin="FusionNode",
        name="fusion_node",
        namespace="",
        parameters=[param_dict],
        extra_arguments=[{"use_intra_process_comms": True}],
    )

    container = ComposableNodeContainer(
        name="wheel_perception_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[fusion_node_plugin],
        output="screen",
    )

    lifecycle_cmd = ExecuteProcess(
        cmd=[
            "bash",
            "-lc",
            (
                "configured=0; "
                "for i in $(seq 1 30); do "
                "  if ros2 lifecycle set /fusion_node configure; then configured=1; break; fi; "
                "  echo \"[lifecycle] waiting for /fusion_node ($i/30)\"; "
                "  sleep 1; "
                "done; "
                "if [ \"$configured\" != \"1\" ]; then "
                "  echo \"[lifecycle] failed to configure /fusion_node\"; "
                "  exit 1; "
                "fi; "
                "ros2 lifecycle set /fusion_node activate"
            ),
        ],
        output="screen",
    )

    return LaunchDescription([
        tf_publisher,
        container,
        TimerAction(period=2.0, actions=[lifecycle_cmd]),
    ])

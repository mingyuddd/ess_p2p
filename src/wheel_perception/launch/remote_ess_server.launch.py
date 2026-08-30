import os

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def _load_params(config_path):
    with open(config_path, "r") as stream:
        yaml_content = yaml.safe_load(stream)

    # params.yaml 顶层是 /**（对任意节点生效），直接取其 ros__parameters
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

    # 目标机部署：跑 ESS 深度推理的服务端节点。
    # 客户端（本机 fusion_node, depth_source=ess_remote）发拼接图 ->
    # 本节点推理 -> 发回深度。复用 params.yaml 的 ess 段（engine_path 等）。
    server_plugin = ComposableNode(
        package="wheel_perception",
        plugin="RemoteEssServer",
        name="remote_ess_server",
        namespace="",
        parameters=[param_dict],
    )

    container = ComposableNodeContainer(
        name="remote_ess_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[server_plugin],
        output="screen",
    )

    return LaunchDescription([container])

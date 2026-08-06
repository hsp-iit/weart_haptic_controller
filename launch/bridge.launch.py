from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = str(
        Path(get_package_share_directory("gelsight_weart_bridge"))
        / "config"
        / "bridge.yaml"
    )

    config_arg = DeclareLaunchArgument(
        "config", default_value=default_config, description="Bridge YAML file"
    )
    bridge = Node(
        package="gelsight_weart_bridge",
        executable="bridge_node",
        name="gelsight_weart_bridge",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )
    return LaunchDescription([config_arg, bridge])

"""Launch the combined controller with its YAML configuration file."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("weart_combined_controller"),
        "config",
        "weart_combined_controller.yaml",
    )

    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=default_config,
        description=(
            "Path to the YAML file with the finger/force-topic mapping and "
            "WEART connection settings."
        ),
    )

    node = Node(
        package="weart_combined_controller",
        executable="combined_controller",
        name="weart_combined_controller",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("config_file")],
    )

    return LaunchDescription([config_file_arg, node])

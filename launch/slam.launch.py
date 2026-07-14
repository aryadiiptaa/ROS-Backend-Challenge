#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    robot_package_share = get_package_share_directory("robot_ros_backend")
    slam_toolbox_package_share = get_package_share_directory("slam_toolbox")
    default_slam_params_file = os.path.join(
        robot_package_share, "config", "slam_toolbox.yaml"
    )
    slam_toolbox_launch_file = os.path.join(
        slam_toolbox_package_share, "launch", "online_async_launch.py"
    )
    use_sim_time = LaunchConfiguration("use_sim_time")
    slam_params_file = LaunchConfiguration("slam_params_file")
    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")
    declare_slam_params_file = DeclareLaunchArgument(
        "slam_params_file", default_value=default_slam_params_file
    )
    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(slam_toolbox_launch_file),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "slam_params_file": slam_params_file,
            "autostart": "true",
            "use_lifecycle_manager": "false",
        }.items(),
    )
    return LaunchDescription(
        [declare_use_sim_time, declare_slam_params_file, slam_toolbox]
    )
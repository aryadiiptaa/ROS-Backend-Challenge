import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

spawn_x = LaunchConfiguration("x")
spawn_y = LaunchConfiguration("y")
spawn_z = LaunchConfiguration("z")
spawn_yaw = LaunchConfiguration("yaw")


def generate_launch_description():
    pkg_name = "robot_ros_backend"
    urdf_file = os.path.join(
        get_package_share_directory(pkg_name), "urdf", "robot.urdf"
    )
    world_file = os.path.join(
        get_package_share_directory(pkg_name), "worlds", "arena.sdf"
    )
    with open(urdf_file, "r") as infp:
        robot_desc = infp.read()
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_desc, "use_sim_time": True}],
    )
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py"
            )
        ),
        launch_arguments={"gz_args": f"-r -v 4 {world_file}"}.items(),
    )
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "challenge_rover",
            "-topic",
            "robot_description",
            "-x",
            spawn_x,
            "-y",
            spawn_y,
            "-z",
            spawn_z,
            "-Y",
            spawn_yaw,
        ],
        output="screen",
    )
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/model/challenge_rover/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        ],
        remappings=[("/model/challenge_rover/tf", "/tf")],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("x", default_value="0.0"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("z", default_value="0.5"),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            robot_state_publisher,
            gazebo,
            spawn_robot,
            ros_gz_bridge,
        ]
    )
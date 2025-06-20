from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_prefix
import os

def generate_launch_description():
    pkg_prefix = get_package_prefix('mppi')

    urdf_file = os.path.join(pkg_prefix, 'share', 'mppi', 'urdf', 'mppi.urdf')
    rviz_config_file = os.path.join(pkg_prefix, 'share', 'mppi', 'rviz', 'default.rviz')
    joint_pub_exec = os.path.join(pkg_prefix, 'lib', 'mppi', 'joint_publisher.py')

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{
                'robot_description': open(urdf_file).read()
            }],
            output='screen'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_file],
            output='screen'
        ),
        ExecuteProcess(
            cmd=['python3', joint_pub_exec],
            output='screen'
        )
    ])

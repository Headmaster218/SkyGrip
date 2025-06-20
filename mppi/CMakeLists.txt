cmake_minimum_required(VERSION 3.5)
project(mppi)

# Python only, no C/C++设置
# 所以以下可以全部去掉：
# set(CMAKE_C_STANDARD 99)
# set(CMAKE_CXX_STANDARD 14)
# add_compile_options()

# 找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# 安装资源文件（urdf, meshes, launch, rviz）
install(DIRECTORY
  urdf
  meshes
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# 安装 Python 脚本（必须 chmod +x）
install(PROGRAMS
  scripts/joint_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

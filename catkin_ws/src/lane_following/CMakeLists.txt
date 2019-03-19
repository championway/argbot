cmake_minimum_required(VERSION 2.8.3)
project(lane_following)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  gazebo_ros
  geometry_msgs
  nav_msgs
  pcl_ros
  pcl_conversions
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  tf_conversions
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
    cfg/pos_PID.cfg
    cfg/ang_PID.cfg
    cfg/lookahead.cfg
)

catkin_package(
)


## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

SET(CMAKE_CXX_FLAGS "-std=c++0x")
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(lidar_detection src/lidar_detection.cpp)
target_link_libraries(lidar_detection ${catkin_LIBRARIES})

add_executable(rgbd_detection src/rgbd_detection.cpp)
target_link_libraries(rgbd_detection ${catkin_LIBRARIES})
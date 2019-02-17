cmake_minimum_required(VERSION 2.8.3)
project(lidar_perception)

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

add_executable(wall_detection src/wall_detection.cpp)
target_link_libraries(wall_detection ${catkin_LIBRARIES})

add_executable(watch_tower src/watch_tower.cpp)
target_link_libraries(watch_tower ${catkin_LIBRARIES})
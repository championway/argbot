# Depth camera package

## Requirements

- ROS kinetic (Ubuntu 16.04)
- PCL (Point Cloud Library)

## Installation

```
$ sudo apt-get install ros-kinetic-pcl-conversions
$ sudo apt-get install ros-kinetic-pcl-ros
```

## How to build
Put this package under [ROS_workspace]/src
```
$ catkin_make --pkg depth_camera(do it in your ROS workspace)
```

## 1. Run the detection node
Using roslaunch:
```
$ roslaunch depth_camera obs_detection.cpp
- By this way, you can use the config/obs_detection.yaml to change your parameters
```
Using rosrun:
```
$ rosrun depth_camera obstacle_detection
- By this way, you will use the default parameters
```

### Rviz
Open the following topics
- /local_map
Note: Remember to chnge your fixed frame to "base_link"
(You could also change the name through the code, it depends on the tf frame of your robot)

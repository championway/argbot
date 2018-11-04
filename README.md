# ARGbot - NCTU ARG

|Branch | Developer |
|-------|--------|
|Master |NCTU ARG Lab|
|devel-david|[David Chen](https://github.com/championway)|

# Requirements

- ROS kinetic (Ubuntu 16.04)
- PCL (Point Cloud Library)

# Hardware

## Sensor


### Laser Scanner
- RPlidar

### Monocular Camera
- Pi camera

### IMU
- 

### Ultrasound sensor
- 

# Installation

## Package

```
$ sudo apt-get install ros-kinetic-gazebo-ros-*
```

# How to build

```
$ cd
$ git clone https://github.com/championway/argbot
$ cd ~/argbot/catkin_ws
$ source /opt/ros/kinetic/setup.bash
$ catkin_make
```
Note:
Do the following everytime as you open new terminals

```
$ cd self-driving-robot/catkin_ws
$ source environment.sh
```

## Localization & Mapping

```
$ roslaunch hector_mapping mapping_no_odom.launch
(Hector SLAM without wheel odometry)
```
To do list:
- [ ] Test gmapping with hector slam 
- [ ] Make sure how turtlebot3 localization works

## Kinematics

### Joystick control

```
```

### Wheel odometry

```
```

## Path Planning

### RRT

```
```

### Pure Pursuit

```
```
## Perception

### Point Cloud

```
```

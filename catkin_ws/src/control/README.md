# Control package

# Requirements

- ROS kinetic (Ubuntu 16.04)

# Installation

## Package

```
$ sudo apt-get install ros-kinetic-dynamic-reconfigure
$ sudo apt-get install ros-kinetic-rqt-gui
```

# How to build
Put this package under [ROS_workspace]/src
```
$ catkin_make (do it in your ROS workspace)
```
Note:
- Build your workspace as long as you change your cfg file.
- The config files are in "cfg" folder
- You can save your parameters as yaml files in rqt-gui interface

# Run the PID control
```
$ rosrun control pid_control.py
```

# Use RQT-GUI interface to tune the PID parameters
```
$ rosrun rqt_gui rqt_gui -s reconfigure
```
Choose the "PID_control" and you should see the following:
- Angular
- Angular_station
- Position
- Position_station

Then you can tune parameters whatever you want
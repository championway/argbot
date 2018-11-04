# hector_slam

See the ROS Wiki for documentation: http://wiki.ros.org/hector_slam

## Hector SLAM without wheel odometry

```
$ roslaunch hector_mapping mapping_no_odom.launch
```
Note: Bad with big turn


## Save map
```
$ rosrun map_server map_saver -f ~/argbot/catkin_ws/src/map/[map_name]
```
Note: Connot read map with map_server (Segmentation fault)

#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace ros;
using namespace std;

class publish_marker{
 public:
  publish_marker(NodeHandle& );
  void publish_callback(const sensor_msgs::PointCloud2ConstPtr);
 private:
  Subscriber sub;
  Publisher marker_pub;
  geometry_msgs::Point p1,p2,p3,p4,p5,p6,p7,p8;
  double t;
  uint32_t shape;
  visualization_msgs::Marker marker;
};
/**********************************
Author: David Chen
Date: 2020/01/01 
Last update: 2020/01/21                                                          
LaserScanner Detection
Subscribe: 
  /scan      (sensor_msgs/LaserScan)
Publish:
  /obstacle_marker      (visualization_msgs/MarkerArray)
  /obstacle_marker_line (visualization_msgs/MarkerArray)
  /cluster_result       (sensor_msgs/PointCloud2)
***********************************/ 
#include <iostream>
#include <vector>
#include <time.h>
#include <string>
#include <math.h>
#include <cmath>
//Ros Lib
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/LaserScan.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <std_srvs/Empty.h>
#include <std_srvs/Trigger.h>
//PCL lib
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
//TF lib
#include <tf_conversions/tf_eigen.h>

#define PI 3.14159
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

class LaserScannerDetection{
private:
  string node_name;

  // For point cloud declare variables
  string frame_id; // "laser_frame" for YDLidar

  // 2D lidar positin (according to ground)
  // We need to transform the coordinate
  float lidar_rotate_x; // (degree)
  float lidar_rotate_y; // (degree)
  float lidar_rotate_z; // (degree)
  float lidar_height; // (meter) pleas add a little bit to avoid large slope
  Eigen::Matrix4f tf_x, tf_y, tf_z, tf;

  // (degree) verticle angle view that robot need to consider
  float angle_max;
  float angle_min;

  int num_threshold;  // to avoid noise
  float slope_threshold;  // slope that robot cannot go down -> 0 ~ -1
  vector< pair<float, int> > vec_slope;  // vector for storing the slope
  vector<geometry_msgs::Point> vec_point;

  visualization_msgs::Marker marker;

  ros::NodeHandle nh;
  ros::Subscriber sub_laser;
  ros::Publisher  pub_pcl;
  ros::Publisher  pub_marker;

  ros::ServiceServer service;

public:
  LaserScannerDetection(ros::NodeHandle&);
  void LidarTransform(void);
  float degree2radius(float);
  void cbLaser(const sensor_msgs::LaserScan::ConstPtr&);
  void data_preprocess(const sensor_msgs::LaserScan::ConstPtr, PointCloudXYZRGB::Ptr);
  bool isCliff(void);
  void marker_init(void);
};

// Constructor
LaserScannerDetection::LaserScannerDetection(ros::NodeHandle &n){
  nh = n;
  node_name = ros::this_node::getName();
  marker_init();

  if(!nh.getParam("lidar_rotate_x", lidar_rotate_x)) lidar_rotate_x = -90;
  if(!nh.getParam("lidar_rotate_y", lidar_rotate_y)) lidar_rotate_y = 90;
  if(!nh.getParam("lidar_rotate_z", lidar_rotate_z)) lidar_rotate_z = 0;
  if(!nh.getParam("lidar_height", lidar_height)) lidar_height = 0.5;

  if(!nh.getParam("angle_max", angle_max)) angle_max = 0;
  if(!nh.getParam("angle_min", angle_min)) angle_min = -90;
  if(!nh.getParam("num_threshold", num_threshold)) num_threshold = 5;
  if(!nh.getParam("slope_threshold", slope_threshold)) slope_threshold = -0.5;


  // Transform lidar coordinate
  LidarTransform();

  ROS_INFO("[%s] Initializing ", node_name.c_str());
  ROS_INFO("[%s] Param [lidar_height] = %f", node_name.c_str(), lidar_height);

  // Publisher
  pub_pcl = nh.advertise<sensor_msgs::PointCloud2> ("/vertical_pcl", 1);
  pub_marker = nh.advertise<visualization_msgs::Marker>("/cliff_point", 1);

  // Subscriber
  sub_laser = nh.subscribe("/scan", 1, &LaserScannerDetection::cbLaser, this);
}

void LaserScannerDetection::LidarTransform(void){
  Eigen::Matrix4f tf_x = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f tf_y = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f tf_z = Eigen::Matrix4f::Identity();

  float theta_x = degree2radius(lidar_rotate_x);
  float theta_y = degree2radius(lidar_rotate_y);
  float theta_z = degree2radius(lidar_rotate_z);

  tf_x(1,1) = cos(theta_x);
  tf_x(1,2) = -sin(theta_x);
  tf_x(2,1) = sin(theta_x);
  tf_x(2,2) = cos(theta_x);

  tf_y(0,0) = cos(theta_y);
  tf_y(0,2) = sin(theta_y);
  tf_y(2,0) = -sin(theta_y);
  tf_y(2,2) = cos(theta_y);

  tf_z(0,0) = cos(theta_z);
  tf_z(0,1) = -sin(theta_z);
  tf_z(1,0) = sin(theta_z);
  tf_z(1,1) = cos(theta_z);
  tf_z(3,2) = lidar_height;

  tf = tf_x*tf_y*tf_z;
}

void LaserScannerDetection::cbLaser(const sensor_msgs::LaserScan::ConstPtr& msg){
  const clock_t t_start = clock();

  frame_id = msg->header.frame_id;
  vec_slope.clear();
  vec_point.clear();
  marker.points.clear();
  
  PointCloudXYZRGB::Ptr pcl_cloud(new PointCloudXYZRGB);
  data_preprocess(msg, pcl_cloud);

  if (isCliff()){
    ROS_INFO("Cliff detected");
  }

  // PCL msg --> ROS pointcloud2 msg
  sensor_msgs::PointCloud2 pcl_msgs;
  pcl::toROSMsg(*pcl_cloud, pcl_msgs);
  pcl_msgs.header = msg->header;
  pub_pcl.publish(pcl_msgs);

  marker.header = msg->header;
  pub_marker.publish(marker);
  
  clock_t t_end = clock();
  // cout << "Data preprocess time taken = " << (t_end-t_start)/(double)(CLOCKS_PER_SEC) << endl;

}

void LaserScannerDetection::data_preprocess(const sensor_msgs::LaserScan::ConstPtr scan_in, PointCloudXYZRGB::Ptr pcl_out){
  int counter = 0;
  for (int i = 0; i < scan_in->ranges.size(); i++){
    float range = scan_in->ranges[i];
    float angle = scan_in->angle_min + (i * scan_in->angle_increment);

    Eigen::Vector4f p(range*cos(angle), range*sin(angle), 0, 1);
    p = p.transpose()*tf;

    if (p[0] == 0 || !isfinite(p[1])) continue;

    pcl::PointXYZRGB point;
    point.x = p[0];
    point.y = p[1];
    point.z = p[2];
    point.r = 0;
    point.g = 0;
    point.b = 255;
    geometry_msgs::Point pt;
    pt.x = p[0];
    pt.y = p[1];
    pt.z = p[2];

    if (angle >= degree2radius(angle_min - lidar_rotate_y) 
      && angle <= degree2radius(angle_max - lidar_rotate_y)){

      float slope = p[2]/p[0];

      // vec_slope.push_back(slope);
      vec_slope.push_back(make_pair(slope, counter));
      vec_point.push_back(pt);

      point.r = 255;
      point.g = 255;
      point.b = 0;

      counter++;
    }

    pcl_out->points.push_back(point);
  }
}

bool LaserScannerDetection::isCliff(void){
  sort(vec_slope.begin(), vec_slope.end());
  bool cliff = false;
  geometry_msgs::Point pt;
  pt.x = 0;
  pt.y = 0;
  pt.z = 0;
  geometry_msgs::Point p;

  if (vec_slope.size()>num_threshold){
    cliff = vec_slope[num_threshold].first < slope_threshold;
    p = vec_point[vec_slope[num_threshold].second];
  }
  else{
    cliff = vec_slope[vec_slope.size()-1].first < slope_threshold;
    p = vec_point[vec_slope[vec_slope.size()-1].second];
  }

  marker.points.push_back(pt);
  marker.points.push_back(p);

  if(cliff){
    marker.color.r = 1.0f;
    marker.color.g = 0.0f;
    marker.color.b = 0.0f;
  }
  else{
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
  }

  return cliff;

  // return vec_slope.size()>num_threshold?
  //         vec_slope[num_threshold] < slope_threshold:
  //         vec_slope[vec_slope.size()-1] < slope_threshold;

}

void LaserScannerDetection::marker_init(){
  marker.ns = "basic_shapes";
  marker.id = 0;
  
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;

  marker.scale.x = 0.03;
  marker.scale.y = 0.03;
  marker.scale.z = 0.03;

  marker.color.r = 0.0f;
  marker.color.g = 1.0f;
  marker.color.b = 0.0f;
  marker.color.a = 1.0;
}

float LaserScannerDetection::degree2radius(float degree){
  return degree*M_PI/180.0;
}


int main (int argc, char** argv)
{
  ros::init (argc, argv, "LaserScannerDetection");
  ros::NodeHandle nh("~");
  LaserScannerDetection pn(nh);
  ros::spin ();
  return 0;
}
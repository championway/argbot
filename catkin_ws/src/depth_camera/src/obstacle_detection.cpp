/**********************************
Author: David Chen
Date: 2019/03/30 
Last update: 2019/03/30
Point Cloud Obstacle Detection
Subscribe: 
  /camera/depth_registered/points      (sensor_msgs/PointCloud2)
Publish:
  /pcl_preprocess       (sensor_msgs/PointCloud2)
  /pcl_points           (geometry_msgs/PoseArray)
  /cluster_result       (sensor_msgs/PointCloud2)
***********************************/ 
#include <iostream>
#include <vector>
#include <array>
#include <time.h>
#include <string>
#include <math.h>
//Ros Lib
#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_srvs/Empty.h>
#include <std_srvs/Trigger.h>
//PCL lib
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
//TF lib
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

class Obstacle_Detection{
private:
  string node_name;
  string robot_frame;
  Eigen::Matrix4f transform_matrix;
  Eigen::Matrix4f transform_matrix_1;

  // Only point cloud in these range will be take into account
  double range_min;
  double range_max;
  double angle_min;
  double angle_max;
  double height_max;
  double height_min;

  // Range of robot itself
  double robot_x_max;
  double robot_x_min;
  double robot_y_max;
  double robot_y_min;
  double robot_z_max;
  double robot_z_min;

  int counts;

  // Define map
  const int map_width = 200;
  const int map_height = 200;
  float map_resolution = 0.5;
  nav_msgs::OccupancyGrid occupancygrid;
  // int** map_array;

  ros::NodeHandle nh;
  ros::Subscriber sub_cloud;
  ros::Subscriber sub_point;
  ros::Publisher  pub_cloud;
  ros::Publisher  pub_points;
  ros::Publisher  pub_map;

  //ros::ServiceClient client
  ros::ServiceServer service;

  tf::TransformListener listener;

public:
  Obstacle_Detection(ros::NodeHandle&);
  void cbCloud(const sensor_msgs::PointCloud2ConstPtr&);
  bool obstacle_srv(std_srvs::Trigger::Request&, std_srvs::Trigger::Response&);
  void pcl_preprocess(const PointCloudXYZRGB::Ptr, PointCloudXYZRGB::Ptr);
  void mapping(const PointCloudXYZRGB::Ptr);
  void map2occupancygrid(float&, float&);
};

Obstacle_Detection::Obstacle_Detection(ros::NodeHandle &n){
  nh = n;
  counts = 0;
  node_name = ros::this_node::getName();

  // map_array = new int*[map_height];
  // for (int i = 0; i < map_height; ++i){
  //   map_array[i] = new int[map_width];
  // }

  robot_frame = "/base_link";
  occupancygrid.header.frame_id = robot_frame;
  occupancygrid.info.resolution = map_resolution;
  occupancygrid.info.width = map_width;
  occupancygrid.info.height = map_height;
  occupancygrid.info.origin.position.x = -map_width*map_resolution/2.;
  occupancygrid.info.origin.position.y = -map_height*map_resolution/2.;

  // rotate -90 degree along x axis
  transform_matrix = Eigen::Matrix4f::Identity();
  float theta = -M_PI/2.0;
  transform_matrix(1,1) = cos(theta);
  transform_matrix(1,2) = -sin(theta);
  transform_matrix(2,1) = sin(theta);
  transform_matrix(2,2) = cos(theta);

  // rotate -90 degree along z axis
  transform_matrix_1 = Eigen::Matrix4f::Identity();
  float theta1 = -M_PI/2.0;
  transform_matrix_1(0,0) = cos(theta1);
  transform_matrix_1(0,1) = -sin(theta1);
  transform_matrix_1(1,0) = sin(theta1);
  transform_matrix_1(1,1) = cos(theta1);

  // Declare defalut parameters
  // Set detecting range
  range_min = 0.0;
  range_max = 30.0;
  angle_min = -180.0;
  angle_max = 180.0;
  height_min = -0.3;
  height_max = 0.5;

  // Set robot range to prevent detecting robot itself as obstacle
  robot_x_max = 0.05;
  robot_x_min = -0.6;
  robot_y_max = 0.1;
  robot_y_min = -0.1;
  robot_z_max = 1.;
  robot_z_min = -1.5;

  //Read yaml file and set costumes parameters
  nh.getParam("range_min", range_min);
  nh.getParam("range_max", range_max);
  nh.getParam("angle_min", angle_min);
  nh.getParam("angle_max", angle_max);
  nh.getParam("height_min", height_min);
  nh.getParam("height_max", height_max);

  nh.getParam("robot_x_max", robot_x_max);
  nh.getParam("robot_x_min", robot_x_min);
  nh.getParam("robot_y_max", robot_y_max);
  nh.getParam("robot_y_min", robot_y_min);
  nh.getParam("robot_z_max", robot_z_max);
  nh.getParam("robot_z_min", robot_z_min);

  service = nh.advertiseService("/obstacle_srv", &Obstacle_Detection::obstacle_srv, this);

  ROS_INFO("[%s] Initializing ", node_name.c_str());
  ROS_INFO("[%s] Param [range_max] = %f, [range_min] = %f", node_name.c_str(), range_max, range_min);
  ROS_INFO("[%s] Param [angle_max] = %f, [angle_min] = %f", node_name.c_str(), angle_max, angle_min);
  ROS_INFO("[%s] Param [height_max] = %f, [height_min] = %f", node_name.c_str(), height_max, height_min);

  // Publisher
  pub_cloud = nh.advertise<sensor_msgs::PointCloud2> ("/pcl_preprocess", 1);
  pub_points = nh.advertise<geometry_msgs::PoseArray> ("/pcl_points", 1);
  pub_map = nh.advertise<nav_msgs::OccupancyGrid> ("/local_map", 1);

  // Subscriber
  sub_cloud = nh.subscribe("/camera/depth_registered/points", 1, &Obstacle_Detection::cbCloud, this);
}

bool Obstacle_Detection::obstacle_srv(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
  res.success = 1;
  res.message = "Call obstacle detection service";
  cout << "Call detection service" << endl;
  return true;
}

void Obstacle_Detection::cbCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
  counts++;
  //return if no cloud data
  if ((cloud_msg->width * cloud_msg->height) == 0 || counts % 3 == 0)
  {
    counts = 0;
    return ;
  }
  const clock_t t_start = clock();
  
  // transfer ros msg to point cloud
  PointCloudXYZRGB::Ptr cloud_in(new PointCloudXYZRGB);
  PointCloudXYZRGB::Ptr cloud_out(new PointCloudXYZRGB);
  pcl::fromROSMsg (*cloud_msg, *cloud_in);

  // transform the point cloud coordinate to let it be same as robot coordinate
  pcl::transformPointCloud(*cloud_in, *cloud_in, transform_matrix);
  pcl::transformPointCloud(*cloud_in, *cloud_in, transform_matrix_1);
  // copyPointCloud(*cloud_in, *cloud_out);

  // Remove out of range points and robot points
  pcl_preprocess(cloud_in, cloud_out);
  mapping(cloud_out);

  // ======= convert cluster pointcloud to points =======
  geometry_msgs::PoseArray pose_array;
  for (size_t i = 0; i < cloud_out->points.size(); i++){
    geometry_msgs::Pose p;
    p.position.x = cloud_out->points[i].x;
    p.position.y = cloud_out->points[i].y;
    p.position.z = cloud_out->points[i].z;
    pose_array.poses.push_back(p);
  }

  pose_array.header = cloud_msg->header;
  pose_array.header.frame_id = robot_frame;
  pub_points.publish(pose_array);
  
  clock_t t_end = clock();
  //cout << "PointCloud preprocess time taken = " << (t_end-t_start)/(double)(CLOCKS_PER_SEC) << endl;

  // Publish point cloud
  sensor_msgs::PointCloud2 pcl_output;
  pcl::toROSMsg(*cloud_out, pcl_output);
  pcl_output.header = cloud_msg->header;
  pcl_output.header.frame_id = robot_frame;
  pub_cloud.publish(pcl_output);
}

void Obstacle_Detection::pcl_preprocess(const PointCloudXYZRGB::Ptr cloud_in, PointCloudXYZRGB::Ptr cloud_out){
  // Remove NaN point
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);

  // Range filter
  float dis, angle = 0;
  int num = 0;
  for (int i=0 ; i < cloud_in->points.size() ; i++)
  {
    dis = cloud_in->points[i].x * cloud_in->points[i].x +
          cloud_in->points[i].y * cloud_in->points[i].y;
    dis = sqrt(dis); //dis = distance between the point and the camera
    angle = atan2f(cloud_in->points[i].y, cloud_in->points[i].x);
    angle = angle * 180 / M_PI; //angle of the point in robot coordinate space

    // using color to check your point cloud coordinate
    /*if(cloud_in->points[i].y >= 0){
      cloud_in->points[i].r = 255;
    }*/

    // if the point is in or out of the range we define
    bool is_in_range =  dis >= range_min && dis <= range_max && 
                        angle >= angle_min && angle <= angle_max &&
                        cloud_in->points[i].z >= height_min && cloud_in->points[i].z <= height_max;
    // if the point is belongs to robot itself
    bool is_robot    =  cloud_in->points[i].x <= robot_x_max && cloud_in->points[i].x >= robot_x_min && 
                        cloud_in->points[i].y <= robot_y_max && cloud_in->points[i].y >= robot_y_min && 
                        cloud_in->points[i].z <= robot_z_max && cloud_in->points[i].z >= robot_z_min;
    if (is_in_range && !is_robot)
    {
      cloud_out->points.push_back(cloud_in->points[i]);
      num ++;
    }
  }

  cloud_out->width = num;
  cloud_out->height = 1;
  cloud_out->points.resize(num);

  // Point cloud noise filter
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
  outrem.setInputCloud(cloud_out);
  outrem.setRadiusSearch(0.8);
  outrem.setMinNeighborsInRadius(2);
  outrem.filter(*cloud_out);
}

void Obstacle_Detection::mapping(const PointCloudXYZRGB::Ptr cloud){
  float x;
  float y;
  occupancygrid.data.clear();

  int map_array[map_height][map_width] = {0};

  for (int i=0 ; i < cloud->points.size() ; i++){
    x = cloud->points[i].x;
    y = cloud->points[i].y;
    map2occupancygrid(x, y);
    if (int(y) < map_height && int(x) < map_width && int(y) >= 0 && int(x) >= 0){
      map_array[int(y)][int(x)] = 100;
    }
  }

  for (int j = 0; j < map_height; j++){
    for (int i = 0; i < map_width; i++){
      occupancygrid.data.push_back(map_array[j][i]);
    }
  }
  pub_map.publish(occupancygrid);
  return;
}

void Obstacle_Detection::map2occupancygrid(float& x, float& y){
  x = int((x - occupancygrid.info.origin.position.x)/map_resolution);
  y = int((y - occupancygrid.info.origin.position.y)/map_resolution);
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "Obstacle_Detection");
  ros::NodeHandle nh("~");
  Obstacle_Detection pn(nh);
  ros::spin ();
  return 0;
}
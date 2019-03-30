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
  /local_map            (nav_msgs/OccupancyGrid)
***********************************/ 
#include <iostream>
#include <vector>
#include <array>
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
  float map_resolution;
  nav_msgs::OccupancyGrid occupancygrid;
  int obs_size;
  int dilating_size;

  ros::NodeHandle nh, pnh;
  ros::Subscriber sub_cloud;
  ros::Subscriber sub_point;
  ros::Publisher  pub_cloud;
  ros::Publisher  pub_points;
  ros::Publisher  pub_map;

  ros::ServiceServer service;

public:
  Obstacle_Detection(ros::NodeHandle&, ros::NodeHandle&);
  void cbCloud(const sensor_msgs::PointCloud2ConstPtr&);
  bool obstacle_srv(std_srvs::Trigger::Request&, std_srvs::Trigger::Response&);
  void pcl_preprocess(const PointCloudXYZRGB::Ptr, PointCloudXYZRGB::Ptr);
  void mapping(const PointCloudXYZRGB::Ptr);
  void map2occupancygrid(float&, float&);
};

Obstacle_Detection::Obstacle_Detection(ros::NodeHandle &n, ros::NodeHandle &pn):
                                       nh(n), pnh(pn){
  counts = 0;
  node_name = ros::this_node::getName();

  // rotate -90 degree along x axis
  transform_matrix = Eigen::Matrix4f::Identity();
  float theta = -M_PI/2.0;
  transform_matrix(1,1) = cos(theta);
  transform_matrix(1,2) = sin(theta);
  transform_matrix(2,1) = -sin(theta);
  transform_matrix(2,2) = cos(theta);
  //Read yaml file and set costumes parameters
  if(!pnh.getParam("range_min", range_min)) range_min=0.0;
  if(!pnh.getParam("range_max", range_max)) range_max=30.0;
  if(!pnh.getParam("angle_min", angle_min)) angle_min = -180.0;
  if(!pnh.getParam("angle_max", angle_max)) angle_max = 180.0;
  if(!pnh.getParam("height_min", height_min)) height_min = -0.3;
  if(!pnh.getParam("height_max", height_max)) height_max = 0.5;
  if(!pnh.getParam("robot_x_max", robot_x_max)) robot_x_max=0.05;
  if(!pnh.getParam("robot_x_min", robot_x_min)) robot_x_min=-0.6;
  if(!pnh.getParam("robot_y_max", robot_y_max)) robot_y_max=0.1;
  if(!pnh.getParam("robot_y_min", robot_y_min)) robot_y_min=-0.1;
  if(!pnh.getParam("robot_z_max", robot_z_max)) robot_z_max=1.;
  if(!pnh.getParam("robot_z_min", robot_z_min)) robot_z_min=-1.5;
  if(!pnh.getParam("robot_frame", robot_frame)) robot_frame="/base_link";
  if(!pnh.getParam("map_resolution", map_resolution)) map_resolution=0.3;
  if(!pnh.getParam("obs_size", obs_size)) obs_size=2;
  if(!pnh.getParam("dilating_size", dilating_size)) dilating_size=4;
  // Parameter information
  ROS_INFO("[%s] Initializing ", node_name.c_str());
  ROS_INFO("[%s] Param [range_max] = %f, [range_min] = %f", node_name.c_str(), range_max, range_min);
  ROS_INFO("[%s] Param [angle_max] = %f, [angle_min] = %f", node_name.c_str(), angle_max, angle_min);
  ROS_INFO("[%s] Param [height_max] = %f, [height_min] = %f", node_name.c_str(), height_max, height_min);
  ROS_INFO("[%s] Param [robot_x_max] = %f, [robot_x_min] = %f", node_name.c_str(), robot_x_max, robot_x_min);
  ROS_INFO("[%s] Param [robot_y_max] = %f, [robot_y_min] = %f", node_name.c_str(), robot_y_max, robot_y_min);
  ROS_INFO("[%s] Param [robot_z_max] = %f, [robot_z_min] = %f", node_name.c_str(), robot_z_max, robot_z_min);
  ROS_INFO("[%s] Param [robot_frame] = %s, [map_resolution] = %f", node_name.c_str(), robot_frame.c_str(), map_resolution);
  ROS_INFO("[%s] Param [obs_size] = %d, [dilating_size] = %d", node_name.c_str(), obs_size, dilating_size);
  // Set map meta data
  occupancygrid.header.frame_id = robot_frame;
  occupancygrid.info.resolution = map_resolution;
  occupancygrid.info.width = map_width;
  occupancygrid.info.height = map_height;
  occupancygrid.info.origin.position.x = -map_width*map_resolution/2.;
  occupancygrid.info.origin.position.y = -map_height*map_resolution/2.;
  // Service
  service = nh.advertiseService("/obstacle_srv", &Obstacle_Detection::obstacle_srv, this);
  // Publisher
  pub_cloud = nh.advertise<sensor_msgs::PointCloud2> ("/pcl_preprocess", 1);
  pub_points = nh.advertise<geometry_msgs::PoseArray> ("/pcl_points", 1);
  pub_map = nh.advertise<nav_msgs::OccupancyGrid> ("/local_map", 1);
  // Subscriber
  sub_cloud = nh.subscribe("/velodyne_points", 1, &Obstacle_Detection::cbCloud, this);
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
  // transfer ros msg to point cloud
  PointCloudXYZRGB::Ptr cloud_in(new PointCloudXYZRGB()); 
  PointCloudXYZ::Ptr cloud_in2(new PointCloudXYZ()); 
  PointCloudXYZRGB::Ptr cloud_out(new PointCloudXYZRGB());
  pcl::fromROSMsg (*cloud_msg, *cloud_in2); 

  // transform the point cloud coordinate to let it be same as robot coordinate
  //pcl::transformPointCloud(*cloud_in2, *cloud_in2, transform_matrix);
  copyPointCloud(*cloud_in2, *cloud_in); 

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
  for(PointCloudXYZRGB::iterator it=cloud_in->begin();
       it != cloud_in->end();it++){
    dis = it->x * it->x + it->y * it->y;
    dis = sqrt(dis);
    angle = atan2f(it->y, it->x);
    angle = angle * 180 / M_PI; //angle of the point in robot coordinate space

    // If the point is in or out of the range we define
    bool is_in_range =  dis >= range_min && dis <= range_max && 
                        angle >= angle_min && angle <= angle_max &&
                        it->z >= height_min && it->z <= height_max;
    // If the point is belongs to robot itself
    bool is_robot    =  it->x <= robot_x_max && it->x >= robot_x_min && 
                        it->y <= robot_y_max && it->y >= robot_y_min && 
                        it->z <= robot_z_max && it->z >= robot_z_min;
    if(is_in_range && !is_robot)
    {
      cloud_out->points.push_back(*it);
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

  for (PointCloudXYZRGB::iterator it = cloud->begin(); it != cloud->end() ; ++it){
    x = it->x;
    y = it->y;
    map2occupancygrid(x, y); 
    if (int(y) < map_height && int(x) < map_width && int(y) >= 0 && int(x) >= 0){
      map_array[int(y)][int(x)] = 100;
    }
  }

  // Map dilating
  for (int j = 0; j < map_height; j++){
    for (int i = 0; i < map_width; i++){
      if (map_array[j][i] == 100){
        for (int m = -dilating_size; m < dilating_size + 1; m++){
          for (int n = -dilating_size; n < dilating_size + 1; n++){
            if(j+m<0 or j+m>=map_height or i+n<0 or i+n>=map_width) continue;
            if (map_array[j+m][i+n] != 100){
              if (m > obs_size || m < -obs_size || n > obs_size || n < -obs_size){
                if (map_array[j+m][i+n] != 80){
                  map_array[j+m][i+n] = 50;
                }
              }
              else{
                if(j+m<0 or j+m>=map_height or i+n<0 or i+n>=map_width) continue;
                map_array[j+m][i+n] = 80;
              }
            }
          }
        }
      }
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
  ros::NodeHandle nh, pnh("~");
  Obstacle_Detection od(nh, pnh);
  ros::spin ();
  return 0;
}

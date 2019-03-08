/**********************************
Author: David Chen
Date: 2019/01/26 
Last update: 2019/01/26                                                           
Point Cloud Wall Detection
Subscribe: 
  /X1/points      (sensor_msgs/PointCloud2)
Publish:
  /obstacle_marker      (visualization_msgs/MarkerArray)
  /obstacle_marker_line (visualization_msgs/MarkerArray)
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
//TF lib
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#define PI 3.14159
using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

class WallDetection{
private:
  string node_name;

  // For point cloud declare variables
  string source_frame;
  string target_frame;
  tf::StampedTransform transformStamped;
  Eigen::Affine3d transform_eigen;

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

  vector< array<double, 2> >boundary_list;

  int counts;

  ros::NodeHandle nh;
  ros::Subscriber sub_cloud;
  ros::Subscriber sub_point;
  ros::Publisher  pub_cloud;
  ros::Publisher  pub_points;

  //ros::ServiceClient client
  ros::ServiceServer service;

  tf::TransformListener listener;

public:
  WallDetection(ros::NodeHandle&);
  void cbCloud(const sensor_msgs::PointCloud2ConstPtr&);
  bool clear_bounary(std_srvs::Trigger::Request&, std_srvs::Trigger::Response&);
  void pcl_preprocess(const PointCloudXYZRGB::Ptr, PointCloudXYZRGB::Ptr);
};

WallDetection::WallDetection(ros::NodeHandle &n){
  nh = n;
  counts = 0;
  node_name = ros::this_node::getName();

  source_frame = "/map";

  range_min = 0.0;
  range_max = 30.0;
  angle_min = -180.0;
  angle_max = 180.0;
  height_min = 0.3;
  height_max = 1.5;

  robot_x_max = 1.0;
  robot_x_min = -1.0;
  robot_y_max = 1.0;
  robot_y_min = -1.0;
  robot_z_max = 1.0;
  robot_z_min = -1.0;

  //Read yaml file
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

  //client = nh.serviceClient<std_srvs::Empty>("/clear_pointcloud_boundary");
  service = nh.advertiseService("/clear_pointcloud_boundary", &WallDetection::clear_bounary, this);

  ROS_INFO("[%s] Initializing ", node_name.c_str());
  ROS_INFO("[%s] Param [range_max] = %f, [range_min] = %f", node_name.c_str(), range_max, range_min);
  ROS_INFO("[%s] Param [angle_max] = %f, [angle_min] = %f", node_name.c_str(), angle_max, angle_min);
  ROS_INFO("[%s] Param [height_max] = %f, [height_min] = %f", node_name.c_str(), height_max, height_min);

  // Publisher
  pub_cloud = nh.advertise<sensor_msgs::PointCloud2> ("/velodyne_points_preprocess", 1);
  pub_points = nh.advertise<geometry_msgs::PoseArray> ("/pcl_points", 1);

  // Subscriber
  sub_cloud = nh.subscribe("/X1/points", 1, &WallDetection::cbCloud, this);
}

bool WallDetection::clear_bounary(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
  res.success = 1;
  res.message = "Clear all boundary points";
  cout << "Clear all boundary points" << endl;
  boundary_list.clear();
  boundary_list.shrink_to_fit();
  return true;
}

void WallDetection::cbCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
  target_frame = cloud_msg->header.frame_id;
  counts++;
  //return if no cloud data
  if ((cloud_msg->width * cloud_msg->height) == 0 || counts % 3 == 0)
  {
    counts = 0;
    return ;
  }
  const clock_t t_start = clock();
  
  // transfer ros msg to point cloud
  PointCloudXYZ::Ptr cloud_XYZ(new PointCloudXYZ);
  PointCloudXYZRGB::Ptr cloud_in(new PointCloudXYZRGB);
  PointCloudXYZRGB::Ptr cloud_out(new PointCloudXYZRGB);
  pcl::fromROSMsg (*cloud_msg, *cloud_XYZ);
  copyPointCloud(*cloud_XYZ, *cloud_in);

  // Remove out of range points and robot points
  pcl_preprocess(cloud_in, cloud_out);

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
  //pose_array.header.frame_id = source_frame;
  pub_points.publish(pose_array);
  
  clock_t t_end = clock();
  //cout << "PointCloud preprocess time taken = " << (t_end-t_start)/(double)(CLOCKS_PER_SEC) << endl;

  // Publish point cloud
  sensor_msgs::PointCloud2 pcl_output;
  pcl::toROSMsg(*cloud_out, pcl_output);
  pcl_output.header = cloud_msg->header;
  //pcl_output.header.frame_id = source_frame;
  pub_cloud.publish(pcl_output);
}

void WallDetection::pcl_preprocess(const PointCloudXYZRGB::Ptr cloud_in, PointCloudXYZRGB::Ptr cloud_out){
  float dis, angle = 0;
  int num = 0;
  for (int i=0 ; i <  cloud_in->points.size() ; i++)
  {
    dis = cloud_in->points[i].x * cloud_in->points[i].x +
          cloud_in->points[i].y * cloud_in->points[i].y;
    dis = sqrt(dis);
    angle = atan2f(cloud_in->points[i].y, cloud_in->points[i].x);
    angle = angle * 180 / 3.1415;
    bool is_in_range =  dis >= range_min && dis <= range_max && 
                        angle >= angle_min && angle <= angle_max &&
                        cloud_in->points[i].z >= height_min && cloud_in->points[i].z <= height_max;
    
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

  /*try{
    listener.waitForTransform(source_frame, target_frame, ros::Time(), ros::Duration(2.0));
    listener.lookupTransform(source_frame, target_frame, ros::Time(), transformStamped);
    tf::transformTFToEigen(transformStamped, transform_eigen);
    pcl::transformPointCloud(*cloud_out, *cloud_out, transform_eigen);
  }
  catch (tf::TransformException ex){
    ROS_INFO("[%s] Can't find transfrom betwen [%s] and [%s] ", node_name.c_str(), source_frame.c_str(), target_frame.c_str());
    return;
  }*/
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "wall_detection");
  ros::NodeHandle nh("~");
  WallDetection pn(nh);
  ros::spin ();
  return 0;
}
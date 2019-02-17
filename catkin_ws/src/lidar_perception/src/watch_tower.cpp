/**********************************
Author: David Chen
Date: 2019/02/26 
Last update: 2019/02/17                                                           
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
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>
//PCL lib
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
//TF lib
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>

#define PI 3.14159
using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;

class WatchTower{
private:
  string node_name;
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

  PointCloudXYZ::Ptr cloud_XYZ{new PointCloudXYZ};
  PointCloudXYZRGB::Ptr cloud_in{new PointCloudXYZRGB};
  PointCloudXYZRGB::Ptr cloud_out{new PointCloudXYZRGB};
  PointCloudXYZRGB::Ptr cloud_cluster{new PointCloudXYZRGB};
  sensor_msgs::PointCloud2 pcl_output;
  nav_msgs::Path a_path;
  nav_msgs::Path b_path;

  visualization_msgs::MarkerArray marker_array;
  geometry_msgs::PoseArray robots;

  bool first = true;
  float past_a_x;
  float past_a_y;
  float past_b_x;
  float past_b_y;

  vector< array<double, 2> >boundary_list;

  int counts;

  ros::NodeHandle nh;
  ros::Subscriber sub_cloud;
  ros::Subscriber sub_point;
  ros::Publisher  pub_cloud;
  ros::Publisher  pub_points;
  ros::Publisher  pub_robot;
  ros::Publisher  pub_path_a;
  ros::Publisher  pub_path_b;
  ros::Publisher  pub_marker;
  ros::Publisher pub_marker_point;

  //ros::ServiceClient client
  ros::ServiceServer service_boundary;
  ros::ServiceServer service_path;

  tf::TransformListener listener;

public:
  WatchTower(ros::NodeHandle&);
  void cbCloud(const sensor_msgs::PointCloud2ConstPtr&);
  void pcl_preprocess(const PointCloudXYZRGB::Ptr, PointCloudXYZRGB::Ptr);
  void cluster_pointcloud(PointCloudXYZRGB::Ptr);
  void drawRviz();
  void tracking(float, float, float, float);
  float distance(float, float, float, float);
  void cbPoint(const geometry_msgs::PoseWithCovarianceStampedPtr&);
  bool pointInBoundary(double, double);
  double product(double, double, double, double);
  void drawBoundary();
  bool clear_bounary(std_srvs::Trigger::Request&, std_srvs::Trigger::Response&);
  bool clear_path(std_srvs::Trigger::Request&, std_srvs::Trigger::Response&);
  void sortBoundary();
};

WatchTower::WatchTower(ros::NodeHandle &n){
  nh = n;
  counts = 0;
  node_name = ros::this_node::getName();

  range_min = 0.0;
  range_max = 40.0;
  angle_min = -180.0;
  angle_max = 180.0;
  height_min = -1.5;
  height_max = 5;

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

  service_boundary = nh.advertiseService("/clear_pointcloud_boundary", &WatchTower::clear_bounary, this);
  service_path = nh.advertiseService("/clear_path", &WatchTower::clear_path, this);

  ROS_INFO("[%s] Initializing ", node_name.c_str());
  ROS_INFO("[%s] Param [range_max] = %f, [range_min] = %f", node_name.c_str(), range_max, range_min);
  ROS_INFO("[%s] Param [angle_max] = %f, [angle_min] = %f", node_name.c_str(), angle_max, angle_min);
  ROS_INFO("[%s] Param [height_max] = %f, [height_min] = %f", node_name.c_str(), height_max, height_min);

  // Publisher
  pub_cloud = nh.advertise<sensor_msgs::PointCloud2> ("/velodyne_points_preprocess", 1);
  pub_points = nh.advertise<geometry_msgs::PoseArray> ("/pcl_points", 1);
  pub_robot = nh.advertise<visualization_msgs::MarkerArray> ("/wamvs", 1);
  pub_path_a = nh.advertise<nav_msgs::Path> ("/path_a", 1);
  pub_path_b = nh.advertise<nav_msgs::Path> ("/path_b", 1);
  pub_marker = nh.advertise< visualization_msgs::Marker >("/boundary_marker", 1);
  pub_marker_point = nh.advertise< visualization_msgs::Marker >("/boundary_marker_point", 1);

  // Subscriber
  sub_cloud = nh.subscribe("/bamboobotb/velodyne_points", 1, &WatchTower::cbCloud, this);
  sub_point = nh.subscribe("/initialpose", 1, &WatchTower::cbPoint, this);
}

bool WatchTower::clear_bounary(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
  res.success = 1;
  res.message = "Clear all boundary points";
  cout << "Clear all boundary points" << endl;
  boundary_list.clear();
  boundary_list.shrink_to_fit();
  return true;
}

bool WatchTower::clear_path(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
  res.success = 1;
  res.message = "Clear all path";
  cout << "Clear all path" << endl;
  a_path.poses.clear();
  b_path.poses.clear();
  first = true;
  return true;
}

void WatchTower::cbCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
  cloud_XYZ->clear();
  cloud_in->clear();
  cloud_out->clear();
  //pcl_output.clear();
  pcl_output.header = cloud_msg->header;
  counts++;
  //return if no cloud data
  if ((cloud_msg->width * cloud_msg->height) == 0 || counts % 3 == 0)
  {
    counts = 0;
    return ;
  }
  const clock_t t_start = clock();
  
  // transfer ros msg to point cloud
  pcl::fromROSMsg (*cloud_msg, *cloud_XYZ);
  copyPointCloud(*cloud_XYZ, *cloud_in);

  // Remove out of range points and robot points
  pcl_preprocess(cloud_in, cloud_out);
  cluster_pointcloud(cloud_out);
  if (robots.poses.size() <= 2){
    if (first){
      first = false;
      past_a_x = robots.poses[0].position.x;
      past_a_y = robots.poses[0].position.y;
      past_b_x = robots.poses[1].position.x;
      past_b_y = robots.poses[1].position.y;
    }
    drawRviz();
    tracking(past_a_x, past_a_y, past_b_x, past_b_y);
  }

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
  pub_points.publish(pose_array);
  
  clock_t t_end = clock();
  //cout << "PointCloud preprocess time taken = " << (t_end-t_start)/(double)(CLOCKS_PER_SEC) << endl;

  // Publish point cloud
  pcl::toROSMsg(*cloud_out, pcl_output);
  pcl_output.header = cloud_msg->header;
  pub_cloud.publish(pcl_output);
}

void WatchTower::pcl_preprocess(const PointCloudXYZRGB::Ptr cloud_in, PointCloudXYZRGB::Ptr cloud_out){
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

    if (is_in_range && !is_robot){
      if(pointInBoundary(cloud_in->points[i].x, cloud_in->points[i].y)){
        cloud_out->points.push_back(cloud_in->points[i]);
        num ++;
      }
      else;
    }
  }

  cloud_out->width = num;
  cloud_out->height = 1;
  cloud_out->points.resize(num);
}

void WatchTower::cluster_pointcloud(PointCloudXYZRGB::Ptr cloud_out){
  
  //========== Outlier remove ==========
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
  // build the filter
  outrem.setInputCloud(cloud_out);
  outrem.setRadiusSearch(1.6);
  outrem.setMinNeighborsInRadius(3);
  // apply filter
  outrem.filter (*cloud_out);
  

  //========== Point Cloud Clustering ==========
  // Declare variable
  int num_cluster = 0;
  int start_index = 0;

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (cloud_out);

  // Create cluster object
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (1.7);// unit: meter
  ec.setMinClusterSize (4);
  ec.setMaxClusterSize (1000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_out);
  ec.extract (cluster_indices);

  robots.poses.clear();

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){
    Eigen::Vector4f centroid;
    cloud_cluster->clear();
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
      cloud_cluster->points.push_back (cloud_out->points[*pit]);
    }
    pcl::compute3DCentroid(*cloud_cluster, centroid);
    geometry_msgs::Pose p;
    p.position.x = centroid[0];
    p.position.y = centroid[1];
    p.position.z = centroid[2];
    robots.poses.push_back(p);
  }

}

void WatchTower::drawRviz(){
  marker_array.markers.clear();
  marker_array.markers.resize(robots.poses.size());
  for (int i = 0; i < robots.poses.size(); i++){
    marker_array.markers[i].header = pcl_output.header;
    marker_array.markers[i].id = i;
    marker_array.markers[i].type = visualization_msgs::Marker::CUBE;
    marker_array.markers[i].action = visualization_msgs::Marker::ADD;
    //marker_array.markers[i].lifetime = ros::Duration(0.5);
    marker_array.markers[i].pose.position.x = robots.poses[i].position.x;
    marker_array.markers[i].pose.position.y = robots.poses[i].position.y;
    marker_array.markers[i].pose.position.z = robots.poses[i].position.z;
    marker_array.markers[i].pose.orientation.x = 0.0;
    marker_array.markers[i].pose.orientation.y = 0.0;
    marker_array.markers[i].pose.orientation.z = 0.0;
    marker_array.markers[i].pose.orientation.w = 1.0;
    marker_array.markers[i].scale.x = 1;
    marker_array.markers[i].scale.y = 1;
    marker_array.markers[i].scale.z = 1;
    marker_array.markers[i].color.r = 0;
    marker_array.markers[i].color.g = 1;
    marker_array.markers[i].color.b = 0;
    marker_array.markers[i].color.a = 1;
  }
  pub_robot.publish(marker_array);
}

void WatchTower::tracking(float x_a, float y_a, float x_b, float y_b){
  a_path.header = pcl_output.header;
  b_path.header = pcl_output.header;
  float dis_a_1 = 10e5;
  float dis_b_1 = 10e5;
  float dis_a_2 = 10e5;
  float dis_b_2 = 10e5;

  bool a_1 = false;
  bool a_2 = false;
  bool b_1 = false;
  bool b_2 = false;

  for(int i = 0; i < robots.poses.size(); i++){
    if(i == 0){
      dis_a_1 = distance(x_a, y_a, robots.poses[i].position.x, robots.poses[i].position.y);
      dis_b_1 = distance(x_b, y_b, robots.poses[i].position.x, robots.poses[i].position.y);
    }
    else if (i == 1){
      dis_a_2 = distance(x_a, y_a, robots.poses[i].position.x, robots.poses[i].position.y);
      dis_b_2 = distance(x_b, y_b, robots.poses[i].position.x, robots.poses[i].position.y);
    }
  }
  if (dis_a_1 < dis_a_2){a_1 = true;}
  else {a_2 = true;}
  if (dis_b_1 < dis_b_2){b_1 = true;}
  else {b_2 = true;}

  if (a_1 && b_1){
    if (dis_a_1 < dis_b_1){
      b_1 = false;
      b_2 = true;
    }
    else{
      a_1 = false;
      a_2 = true;
    }
  }

  else if (a_2 && b_2){
    if (dis_a_2 < dis_b_2){
      b_2 = false;
      b_1 = true;
    }
    else{
      a_2 = false;
      a_1 = true;
    }
  }
  geometry_msgs::PoseStamped p0;
  geometry_msgs::PoseStamped p1;
  p0.pose = robots.poses[0];
  p1.pose = robots.poses[1];
  if (a_1){
    if (dis_a_1 != 10e5 && dis_a_1 < 7){
      a_path.poses.push_back(p0);
      past_a_x = p0.pose.position.x;
      past_a_y = p0.pose.position.y;
    }
  }
  else if (a_2){
    if (dis_a_2 != 10e5 && dis_a_2 < 12){
      a_path.poses.push_back(p1);
      past_a_x = p1.pose.position.x;
      past_a_y = p1.pose.position.y;
    }
  }

  if (b_1){
    if (dis_b_1 != 10e5 && dis_b_1 < 12){
      b_path.poses.push_back(p0);
      past_b_x = p0.pose.position.x;
      past_b_y = p0.pose.position.y;
    }
  }
  else if (b_2){
    if (dis_b_2 != 10e5 && dis_b_2 < 12){
      b_path.poses.push_back(p1);
      past_b_x = p1.pose.position.x;
      past_b_y = p1.pose.position.y;
    }
  }
  pub_path_a.publish(a_path);
  pub_path_b.publish(b_path);
}

float WatchTower::distance(float x1, float y1, float x2, float y2){
  return sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
}

void WatchTower::cbPoint(const geometry_msgs::PoseWithCovarianceStampedPtr& point_msg){
  std::cout<< "Get boundary point" << std::endl;
  float x = point_msg->pose.pose.position.x;
  float y = point_msg->pose.pose.position.y;
  array<double, 2> point= {x, y};
  boundary_list.push_back(point);
  sortBoundary();
  
  // Draw the boudary area
  drawBoundary();
}

void WatchTower::sortBoundary(){
  double center_x = 0;
  double center_y = 0;
  for (int i=0; i< boundary_list.size() ;i++){
    array<double, 2>base = boundary_list[i];
    center_x += base.at(0);
    center_y += base.at(1);
  }
  center_x = center_x / (boundary_list.size());
  center_y = center_y / (boundary_list.size());
  vector< array<double, 2> >sort_boundary;
  int *already_find = new int[boundary_list.size()];
  for (int i = 0; i < boundary_list.size(); ++i){
      already_find[i] = 0;
  }

  for (int i=0; i< boundary_list.size() ;i++){
    double max_angle = -100, x, y, angle;
    int index = 0;
    for(int j=0;j<boundary_list.size();j++){
      x = boundary_list[j].at(0) - center_x;
      y = boundary_list[j].at(1) - center_y;
      double angle = (atan2(y, x) / PI * 180 + 360);
      angle = fmod(angle, 360);
      if(angle > max_angle && already_find[j]==0 ){
        max_angle = angle;
        index = j;
      }
    }
    sort_boundary.push_back(boundary_list[index]);
    already_find[index] += 1;
  }
  boundary_list = sort_boundary;
  delete already_find;
}

void WatchTower::drawBoundary(){
  /*
  ***************************************************************
    Draw boundary area
  ***************************************************************
  */
  if (boundary_list.size() <= 0)
    return ;
  visualization_msgs::Marker points;
  points.id = 5;
  points.type = visualization_msgs::Marker::POINTS;
  points.action = visualization_msgs::Marker::ADD;
  points.scale.x = 1.0;
  points.scale.y = 1.0;
  points.color.g = 1.0f;
  points.color.a = 1.0;
  points.pose.orientation.w = 1.0;
  points.ns = "points_and_lines";
  points.header.stamp = ros::Time::now();
  points.header.frame_id = "/velodyne";
  double x = 0;
  double y = 0;
  for (int i=0; i<= boundary_list.size()-1 ;i++){
    geometry_msgs::Point p1;
    p1.x = boundary_list[i].at(0);
    p1.y = boundary_list[i].at(1);
    x += p1.x;
    y += p1.y;
    p1.z = 0;
    points.points.push_back(p1);
    //cout << p1.x << " " << p1.y << endl;
  }
  x = x / boundary_list.size();
  y = y /boundary_list.size();
  //cout << "===" << endl;
  geometry_msgs::Point tmp;
  tmp.x = x;
  tmp.y = y;
  //points.points.push_back(tmp);
  pub_marker_point.publish(points);

  if (boundary_list.size() <= 2)
    return ;
  
  visualization_msgs::Marker line_list;
  line_list.id = 100;
  line_list.header.stamp = ros::Time::now();
  line_list.header.frame_id = "/velodyne";
  line_list.pose.orientation.w = 1.0;
  line_list.ns = "points_and_lines";
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.scale.x = 0.1;
  line_list.color.a = 1.0;
  line_list.color.r = 1.0;
  for (int i=0; i<= boundary_list.size()-1 ;i++){
    geometry_msgs::Point p1, p2;
    p1.x = boundary_list[i].at(0);
    p1.y = boundary_list[i].at(1);
    p1.z = 0;
    //cout << p1.x << " " << p1.y << endl;

    if(i == boundary_list.size()-1){
      p2.x = boundary_list[0].at(0);
      p2.y = boundary_list[0].at(1);
      p2.z = 0;
    }
    else{
      p2.x = boundary_list[i+1].at(0);
      p2.y = boundary_list[i+1].at(1);
      p2.z = 0;
    }
    line_list.points.push_back(p1);
    line_list.points.push_back(p2);
  }
  pub_marker.publish(line_list);
}

bool WatchTower::pointInBoundary(double x, double y){
  /*
  ***************************************************************
    Using product to confirm the points are inside the boundary area
  ***************************************************************
  */
    if (boundary_list.size() < 3){
        return true;
    }
  bool inside = true;
  for (int i=0; i< boundary_list.size() ;i++){
    array<double ,2>pre;
    array<double, 2>post = {boundary_list[i].at(0)-x, boundary_list[i].at(1)-y};
    if(i == 0){
      pre = {boundary_list[ boundary_list.size()-1 ].at(0)-x, boundary_list[ boundary_list.size()-1 ].at(1)-y};
    }
    else{
      pre = {boundary_list[ i-1 ].at(0)-x, boundary_list[ i-1 ].at(1)-y};
    }
    
    if(product(pre[0], pre[1], post[0], post[1]) > 0.0){
      inside = false;
      break;
    }
  }
  return inside;
}

double WatchTower::product(double v1_x, double v1_y, double v2_x, double v2_y){
  return (v1_x*v2_y - v2_x*v1_y);
}

int main (int argc, char** argv)
{
  ros::init (argc, argv, "WatchTower_tracking");
  ros::NodeHandle nh("~");
  WatchTower pn(nh);
  ros::spin ();
  return 0;
}
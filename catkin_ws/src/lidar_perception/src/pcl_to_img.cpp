/**********************************
Author: David Chen
Date: 2018/06/01 
Last update: 2018/09/10
Point Cloud 
Subscribe: 
  /velodyne_points      (sensor_msgs/PointCloud2)
Publish:
  /obstacle_list        (robotx_msgs/ObstaclePoseList)
  /obj_list             (robotx_msgs/ObjectPoseList)
  /obstacle_marker      (visualization_msgs/MarkerArray)
  /obstacle_marker_line (visualization_msgs/MarkerArray)
  /cluster_result       (sensor_msgs/PointCloud2)
  /pcl_points           (robotx_msgs/PCL_points)
***********************************/ 
#include <ros/ros.h>
#include <cmath>        // std::abs
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/io/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>

//define point cloud type
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;
//typedef boost::shared_ptr <robotx_msgs::BoolStamped const> BoolStampedConstPtr;
//declare point cloud
PointCloudXYZ::Ptr cloud_inXYZ (new PointCloudXYZ);
PointCloudXYZRGB::Ptr cloud_in (new PointCloudXYZRGB); 
PointCloudXYZRGB::Ptr cloud_filtered (new PointCloudXYZRGB);
PointCloudXYZRGB::Ptr plane_filtered (new PointCloudXYZRGB);
PointCloudXYZRGB::Ptr cloud_h (new PointCloudXYZRGB);
PointCloudXYZRGB::Ptr cloud_f (new PointCloudXYZRGB);
PointCloudXYZRGB::Ptr cloud_plane (new PointCloudXYZRGB);
PointCloudXYZRGB::Ptr cloud_scene(new PointCloudXYZRGB);
//PointCloudXYZRGB::Ptr wall (new PointCloudXYZRGB);
PointCloudXYZRGB::Ptr result (new PointCloudXYZRGB);
sensor_msgs::PointCloud2 ros_out;
sensor_msgs::PointCloud2 ros_cluster;

//declare ROS publisher
ros::Publisher pub_points;

//declare global variable 
bool lock = false;
ros::Time pcl_t;
float region = 60;

//declare function
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr&); //point cloud subscriber call back function
void pcl2posearray(void); //point cloud clustering

void callback(const sensor_msgs::PointCloud2ConstPtr& input)
{
  if (!lock){
    lock = true;
    //covert from ros type to pcl type
    pcl_t = input->header.stamp;
    pcl::fromROSMsg (*input, *cloud_inXYZ);
    copyPointCloud(*cloud_inXYZ, *cloud_in);
    pcl2posearray();
    std::cout<< "Publish" << std::endl;
  }
  else{
    std::cout << "lock" << std::endl;
  }
}

//void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
void pcl2posearray()
{
  //std::cout<< "start processing point clouds" << std::endl;

  copyPointCloud(*cloud_in, *cloud_filtered);
  //========== Downsample ==========
  pcl::VoxelGrid<pcl::PointXYZRGB> vg;
  vg.setInputCloud (cloud_in);
  vg.setLeafSize (0.1f, 0.1f, 0.1f); //unit:cetimeter
  vg.filter (*cloud_in);

  //========== Outlier remove ==========
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
  // build the filter
  outrem.setInputCloud(cloud_filtered);
  outrem.setRadiusSearch(0.5);
  outrem.setMinNeighborsInRadius(2);
  // apply filter
  outrem.filter (*cloud_filtered);

  //========== Remove higer and lower place ==========
  pcl::ExtractIndices<pcl::PointXYZRGB> extract_h_l_place;
  pcl::PointIndices::Ptr hl_indices (new pcl::PointIndices);
  for (int i = 0; i < cloud_in->points.size(); i++)
  {
    if (cloud_in->points[i].z <= -1.3)
    {
      hl_indices->indices.push_back(i);
    }
  }
  extract_h_l_place.setInputCloud(cloud_in);
  extract_h_l_place.setIndices(hl_indices);
  extract_h_l_place.setNegative(true);
  extract_h_l_place.filter(*cloud_h);
  *cloud_in = *cloud_h;
  cloud_h->clear();

  //========== Remove WAM-V Region ==========
  pcl::ExtractIndices<pcl::PointXYZRGB> extract_WAMV;
  pcl::PointIndices::Ptr wamv_indices (new pcl::PointIndices);
  for (int i = 0; i < cloud_in->points.size(); i++)
  {
    if(cloud_in->points[i].y <=3.5 && cloud_in->points[i].y >= -3.5 && cloud_in->points[i].x >= -1.5 && cloud_in->points[i].x <= 1.5)
    {
      wamv_indices->indices.push_back(i);
    }
  }
  extract_WAMV.setInputCloud(cloud_in);
  extract_WAMV.setIndices(wamv_indices);
  extract_WAMV.setNegative(true);
  extract_WAMV.filter(*cloud_h);
  *cloud_in = *cloud_h;
  cloud_h->clear();

  //========== Restrict to certain region ==========
  pcl::ExtractIndices<pcl::PointXYZRGB> extract_too_far;
  pcl::PointIndices::Ptr far_indices (new pcl::PointIndices);
  for (int i = 0; i < cloud_in->points.size(); i++)
  {
    if((cloud_in->points[i].x*cloud_in->points[i].x + cloud_in->points[i].y*cloud_in->points[i].y) >= region*region)
    {
      far_indices->indices.push_back(i);
    }
  }
  extract_too_far.setInputCloud(cloud_in);
  extract_too_far.setIndices(far_indices);
  extract_too_far.setNegative(true);
  extract_too_far.filter(*cloud_h);
  *cloud_in = *cloud_h;
  cloud_h->clear();


  // ======= convert cluster pointcloud to points =======
  geometry_msgs::PoseArray pose_arr;
  for (size_t i = 0; i < cloud_in->points.size(); i++){
      geometry_msgs::Pose p;
      p.position.x = cloud_in->points[i].x;
      p.position.y = cloud_in->points[i].y;
      p.position.z = cloud_in->points[i].z;
      pose_arr.poses.push_back(p);
  }
  pose_arr.header.stamp = pcl_t;
  pub_points.publish(pose_arr);
  lock = false;
}


int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "pcl_to_img");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 1, callback);
  // Create a ROS publisher for the output point cloud
  pub_points = nh.advertise<geometry_msgs::PoseArray> ("/pcl_points_img", 1);
  ros::spin ();
}
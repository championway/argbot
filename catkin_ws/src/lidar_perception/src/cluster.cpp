/**********************************
Author: David Chen
Date: 2018/06/01 
Last update: 2018/07/22                                                              
Point Cloud Clustering
Subscribe: 
  /velodyne_points      (sensor_msgs/PointCloud2)
Publish:
  /obstacle_list        (robotx_msgs/ObstaclePoseList)
  /obj_list          (robotx_msgs/ObjectPoseList)
  /obstacle_marker      (visualization_msgs/MarkerArray)
  /obstacle_marker_line (visualization_msgs/MarkerArray)
  /cluster_result       (sensor_msgs/PointCloud2)
  /pcl_points           (robotx_msgs/PCL_points)
***********************************/ 
#include <ros/ros.h>
#include <cmath>        // std::abs
#include <sensor_msgs/PointCloud2.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/io/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <robotx_msgs/ObstaclePose.h>
#include <robotx_msgs/ObstaclePoseList.h>
#include <robotx_msgs/PCL_points.h>
#include <robotx_msgs/ObjectPose.h>
#include <robotx_msgs/ObjectPoseList.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <std_msgs/ColorRGBA.h>
#include <std_msgs/Time.h>
#include <std_msgs/String.h>
#include <tf/transform_listener.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <Eigen/Dense>
#include <pcl/filters/project_inliers.h>

using namespace Eigen;
using namespace message_filters;
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
ros::Publisher pub_result;
ros::Publisher pub_marker;
ros::Publisher pub_marker_line;
ros::Publisher pub_obstacle;
ros::Publisher pub_object;
ros::Publisher pub_points;

//declare global variable
std_msgs::String pcl_frame_id; 
bool lock = false;
float low = -0.25;
float high = 1.5-low;
float thres_low = 0.03;
float thres_high = 1.5;
float feature_sampling_space = 0.1;
bool  visual;
visualization_msgs::MarkerArray marker_array;
visualization_msgs::MarkerArray marker_array_line;
ros::Time pcl_t;

//declare function
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr&); //point cloud subscriber call back function
void cluster_pointcloud(void); //point cloud clustering
void drawRviz(robotx_msgs::ObstaclePoseList); //draw marker in Rviz
void drawRviz_line(robotx_msgs::ObstaclePoseList); //draw marker line list in Rviz

void callback(const sensor_msgs::PointCloud2ConstPtr& input)
{
  if (!lock){
    lock = true;
    //covert from ros type to pcl type
    pcl_frame_id.data = input->header.frame_id;
    pcl_t = input->header.stamp;
    pcl::fromROSMsg (*input, *cloud_inXYZ);
    copyPointCloud(*cloud_inXYZ, *cloud_in);

    //set color for point cloud
    for (size_t i = 0; i < cloud_in->points.size(); i++){
      cloud_in->points[i].r = 255;
      cloud_in->points[i].g = 255;
      cloud_in->points[i].b = 0;
    }
    clock_t t_start = clock();
    cluster_pointcloud();
    clock_t t_end = clock();
    //std::cout << "Pointcloud cluster time taken = " << (t_end-t_start)/(double)(CLOCKS_PER_SEC) << std::endl;
  }
  else{
    std::cout << "lock" << std::endl;
  }
}

//void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
void cluster_pointcloud()
{
  //std::cout<< "start processing point clouds" << std::endl;
  

  copyPointCloud(*cloud_in, *cloud_filtered);
  //========== Remove NaN point ==========
  /*std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);*/

  //========== Outlier remove ==========
  /*pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> out_filter;
  out_filter.setInputCloud (cloud_filtered);
  out_filter.setMeanK (50);
  out_filter.setStddevMulThresh (1.0);
  out_filter.filter (*cloud_filtered);*/

  //========== Remove higer and lower place ==========
  /*pcl::ExtractIndices<pcl::PointXYZRGB> extract_h_l_place;
  pcl::PointIndices::Ptr hl_indices (new pcl::PointIndices);
  for (int i = 0; i < cloud_filtered->points.size(); i++)
  {
    if (cloud_filtered->points[i].z >= high || cloud_filtered->points[i].z <= low)
    {
      hl_indices->indices.push_back(i);
    }
  }
  extract_h_l_place.setInputCloud(cloud_filtered);
  extract_h_l_place.setIndices(hl_indices);
  extract_h_l_place.setNegative(true);
  extract_h_l_place.filter(*cloud_h);
  *cloud_filtered = *cloud_h;*/

  //========== Remove WAM-V Region ==========
  
  pcl::ExtractIndices<pcl::PointXYZRGB> extract_WAMV;
  pcl::PointIndices::Ptr hl_indices (new pcl::PointIndices);
  for (int i = 0; i < cloud_filtered->points.size(); i++)
  {
    if(cloud_filtered->points[i].y <= 3.2 && cloud_filtered->points[i].y >= -3.5 && cloud_filtered->points[i].x >= -1.5 && cloud_filtered->points[i].x <= 1.5)
    {
      hl_indices->indices.push_back(i);
    }
  }
  extract_WAMV.setInputCloud(cloud_filtered);
  extract_WAMV.setIndices(hl_indices);
  extract_WAMV.setNegative(true);
  extract_WAMV.filter(*cloud_h);
  *cloud_filtered = *cloud_h;
  

  //========== Downsample ==========
  /*pcl::VoxelGrid<pcl::PointXYZRGB> vg;
  vg.setInputCloud (cloud_filtered);
  vg.setLeafSize (0.08f, 0.08f, 0.08f); //unit:cetimeter
  vg.filter (*cloud_filtered);*/

  //========== Point Cloud Clustering ==========
  // Declare variable
  int num_cluster = 0;
  int start_index = 0;
  robotx_msgs::ObstaclePoseList ob_list;
  robotx_msgs::ObjectPoseList obj_list;
  robotx_msgs::PCL_points pcl_points;

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (cloud_filtered);

  // Create cluster object
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (2.2);// unit: meter
  ec.setMinClusterSize (3);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);
  
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    // Declare variable
    float x_min_x = 10e5;
    float x_min_y = 10e5;
    float y_min_x = 10e5;
    float y_min_y = 10e5;
    float x_max_x = -10e5;
    float x_max_y = -10e5;
    float y_max_x = -10e5;
    float y_max_y = -10e5; 
    robotx_msgs::ObstaclePose ob_pose;
    robotx_msgs::ObjectPose obj_pose;
    Eigen::Vector4f centroid;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
      result->points.push_back(cloud_filtered->points[*pit]);
      if (cloud_filtered->points[*pit].x < x_min_x)
      {
        x_min_x = cloud_filtered->points[*pit].x;
        x_min_y = cloud_filtered->points[*pit].y;
      }
      if (cloud_filtered->points[*pit].x > x_max_x)
      {
        x_max_x = cloud_filtered->points[*pit].x;
        x_max_y = cloud_filtered->points[*pit].y;
      }
      if (cloud_filtered->points[*pit].y < y_min_y)
      {
        y_min_x = cloud_filtered->points[*pit].x;
        y_min_y = cloud_filtered->points[*pit].y;
      }
      if (cloud_filtered->points[*pit].y > y_max_y)
      {
        y_max_x = cloud_filtered->points[*pit].x;
        y_max_y = cloud_filtered->points[*pit].y;
      }
    }
    
    num_cluster++;
    // ======= convert cluster pointcloud to points =======
    geometry_msgs::PoseArray pose_arr;
    for (size_t i = 0; i < cloud_cluster->points.size(); i++){
        geometry_msgs::Pose p;
        p.position.x = cloud_cluster->points[i].x;
        p.position.y = cloud_cluster->points[i].y;
        p.position.z = cloud_cluster->points[i].z;
        pose_arr.poses.push_back(p);
    }
    pcl_points.list.push_back(pose_arr);

    // ======= add cluster centroid =======
    pcl::compute3DCentroid(*cloud_cluster, centroid);
    geometry_msgs::Point c;
    c.x = centroid[0];
    c.y = centroid[1];
    c.z = centroid[2];
    pcl_points.centroids.push_back(c);
    //pub_PoseArray.publish(pose_arr);
    //pub_2d_pcl.publish(*cloud);

    pcl::toROSMsg(*cloud_cluster, ros_cluster);

    obj_pose.header.stamp = pcl_t;
    //obj_pose.header.stamp = ros::Time::now();
    obj_pose.header.frame_id = cloud_in->header.frame_id;
    obj_pose.position.x = centroid[0];
    obj_pose.position.y = centroid[1];
    obj_pose.position.z = centroid[2];
    obj_pose.cloud = ros_cluster;
    obj_list.list.push_back(obj_pose);

    ob_pose.header.stamp = pcl_t;
    //ob_pose.header.stamp = ros::Time::now();
    ob_pose.header.frame_id = cloud_in->header.frame_id;
    ob_pose.cloud = ros_cluster;
    ob_pose.x = centroid[0];
    ob_pose.y = centroid[1];
    ob_pose.z = centroid[2];
    Eigen::Vector4f min;
    Eigen::Vector4f max;
    pcl::getMinMax3D (*cloud_cluster, min, max);
    ob_pose.min_x = min[0];
    ob_pose.max_x = max[0];
    ob_pose.min_y = min[1];
    ob_pose.max_y = max[1];
    ob_pose.min_z = min[2];
    ob_pose.max_z = max[2];
    ob_pose.x_min_x = x_min_x;
    ob_pose.x_min_y = x_min_y;
    ob_pose.x_max_x = x_max_x;
    ob_pose.x_max_y = x_max_y;
    ob_pose.y_min_x = y_min_x;
    ob_pose.y_min_y = y_min_y;
    ob_pose.y_max_x = y_max_x;
    ob_pose.y_max_y = y_max_y;
    //ob_pose.r = 1;
    ob_list.list.push_back(ob_pose);
    start_index = result->points.size();
  }

  //set obstacle list
  obj_list.header.stamp = pcl_t;
  //obj_list.header.stamp = ros::Time::now();
  obj_list.header.frame_id = cloud_in->header.frame_id;
  obj_list.size = num_cluster;
  pub_object.publish(obj_list);

  ob_list.header.stamp = pcl_t;
  //ob_list.header.stamp = ros::Time::now();
  ob_list.header.frame_id = cloud_in->header.frame_id;
  ob_list.size = num_cluster;
  pub_obstacle.publish(ob_list);

  pcl_points.header.stamp = pcl_t;
  //pcl_points.header.stamp = ros::Time::now();
  pcl_points.header.frame_id = cloud_in->header.frame_id;
  pub_points.publish(pcl_points);
  if (visual){
    drawRviz(ob_list);
    drawRviz_line(ob_list);
  }
  result->header.frame_id = cloud_in->header.frame_id;
  pcl::toROSMsg(*result, ros_out);
  ros_out.header.stamp = pcl_t;
  //ros_out.header.stamp = ros::Time::now();
  pub_result.publish(ros_out);
  lock = false;
  result->clear();
  //std::cout << "Finish" << std::endl << std::endl; 
}

void drawRviz_line(robotx_msgs::ObstaclePoseList ob_list){
  marker_array_line.markers.resize(ob_list.size);
  for (int i = 0; i < ob_list.size; i++)
  {
    marker_array_line.markers[i].header.frame_id = pcl_frame_id.data;
    marker_array_line.markers[i].id = i;
    marker_array_line.markers[i].header.stamp = ob_list.header.stamp;
    marker_array_line.markers[i].type = visualization_msgs::Marker::LINE_STRIP;
    marker_array_line.markers[i].action = visualization_msgs::Marker::ADD;
    //marker_array.markers[i].pose.orientation.w = 1.0;
    marker_array_line.markers[i].points.clear();
    marker_array_line.markers[i].lifetime = ros::Duration(0.5);
    marker_array_line.markers[i].scale.x = (0.1);
    geometry_msgs::Point x_min;
    x_min.x = ob_list.list[i].x_min_x;
    x_min.y = ob_list.list[i].x_min_y;
    geometry_msgs::Point x_max;
    x_max.x = ob_list.list[i].x_max_x;
    x_max.y = ob_list.list[i].x_max_y;
    geometry_msgs::Point y_min;
    y_min.x = ob_list.list[i].y_min_x;
    y_min.y = ob_list.list[i].y_min_y;
    geometry_msgs::Point y_max;
    y_max.x = ob_list.list[i].y_max_x;
    y_max.y = ob_list.list[i].y_max_y;
    marker_array_line.markers[i].points.push_back(x_min);
    marker_array_line.markers[i].points.push_back(y_min);
    marker_array_line.markers[i].points.push_back(x_max);
    marker_array_line.markers[i].points.push_back(y_max);
    marker_array_line.markers[i].points.push_back(x_min);
    if (ob_list.list[i].r == 1)
    {
      marker_array_line.markers[i].text = "Buoy";
      marker_array_line.markers[i].color.r = 0;
      marker_array_line.markers[i].color.g = 0;
      marker_array_line.markers[i].color.b = 1;
      marker_array_line.markers[i].color.a = 1;
    }
    else if (ob_list.list[i].r == 2)
    {
      marker_array_line.markers[i].text = "Totem";
      marker_array_line.markers[i].color.r = 0;
      marker_array_line.markers[i].color.g = 1;
      marker_array_line.markers[i].color.b = 0;
      marker_array_line.markers[i].color.a = 1;
    }
    else if (ob_list.list[i].r == 3)
    {
      marker_array_line.markers[i].text = "Dock";
      marker_array_line.markers[i].color.r = 1;
      marker_array_line.markers[i].color.g = 1;
      marker_array_line.markers[i].color.b = 1;
      marker_array_line.markers[i].color.a = 1;
    }
    else
    {
      marker_array_line.markers[i].color.r = 1;
      marker_array_line.markers[i].color.g = 0;
      marker_array_line.markers[i].color.b = 0;
      marker_array_line.markers[i].color.a = 1;
    }
  }
  pub_marker_line.publish(marker_array_line);
}

void drawRviz(robotx_msgs::ObstaclePoseList ob_list){
      marker_array.markers.resize(ob_list.size);
      std_msgs::ColorRGBA c;
      for (int i = 0; i < ob_list.size; i++)
      {
        marker_array.markers[i].header.frame_id = pcl_frame_id.data;
        marker_array.markers[i].id = i;
        marker_array.markers[i].header.stamp = ob_list.header.stamp;
        marker_array.markers[i].type = visualization_msgs::Marker::CUBE;
        marker_array.markers[i].action = visualization_msgs::Marker::ADD;
        marker_array.markers[i].lifetime = ros::Duration(0.5);
        marker_array.markers[i].pose.position.x = ob_list.list[i].x;
        marker_array.markers[i].pose.position.y = ob_list.list[i].y;
        marker_array.markers[i].pose.position.z = ob_list.list[i].z;
        marker_array.markers[i].pose.orientation.x = 0.0;
        marker_array.markers[i].pose.orientation.y = 0.0;
        marker_array.markers[i].pose.orientation.z = 0.0;
        marker_array.markers[i].pose.orientation.w = 1.0;
        marker_array.markers[i].scale.x = 1;
        marker_array.markers[i].scale.x = 1;
        marker_array.markers[i].scale.x = 1;
        //marker_array.markers[i].scale.x = (ob_list.list[i].max_x-ob_list.list[i].min_x);
        //marker_array.markers[i].scale.y = (ob_list.list[i].max_y-ob_list.list[i].min_y);
        //marker_array.markers[i].scale.z = (ob_list.list[i].max_z-ob_list.list[i].min_z);
        if (marker_array.markers[i].scale.x ==0)
          marker_array.markers[i].scale.x=1;

        if (marker_array.markers[i].scale.y ==0)
          marker_array.markers[i].scale.y=1;

        if (marker_array.markers[i].scale.z ==0)
          marker_array.markers[i].scale.z=1;
        if (ob_list.list[i].r == 1)
        {
          marker_array.markers[i].text = "Buoy";
          marker_array.markers[i].color.r = 0;
          marker_array.markers[i].color.g = 0;
          marker_array.markers[i].color.b = 1;
          marker_array.markers[i].color.a = 0.5;
        }
        else if (ob_list.list[i].r == 2)
        {
          marker_array.markers[i].text = "Totem";
          marker_array.markers[i].color.r = 0;
          marker_array.markers[i].color.g = 1;
          marker_array.markers[i].color.b = 0;
          marker_array.markers[i].color.a = 0.5;
        }
        else if (ob_list.list[i].r == 3)
        {
          marker_array.markers[i].text = "Dock";
          marker_array.markers[i].color.r = 1;
          marker_array.markers[i].color.g = 1;
          marker_array.markers[i].color.b = 1;
          marker_array.markers[i].color.a = 0.5;
        }
        else
        {
          marker_array.markers[i].color.r = 1;
          marker_array.markers[i].color.g = 0;
          marker_array.markers[i].color.b = 0;
          marker_array.markers[i].color.a = 0.5;
        }
      }
      pub_marker.publish(marker_array);
}

int main (int argc, char** argv)
{
  // Initialize ROS
  ros::init (argc, argv, "cluster_extraction");
  ros::NodeHandle nh("~");
  tf::TransformListener listener(ros::Duration(1.0));
  visual = nh.param("visual", false);

  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 1, callback);
  // Create a ROS publisher for the output point cloud
  pub_obstacle = nh.advertise< robotx_msgs::ObstaclePoseList > ("/obstacle_list", 1);
  pub_object = nh.advertise< robotx_msgs::ObjectPoseList > ("/obj_list", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("/obstacle_marker", 1);
  pub_marker_line = nh.advertise<visualization_msgs::MarkerArray>("/obstacle_marker_line", 1);
  pub_result = nh.advertise<sensor_msgs::PointCloud2> ("/cluster_result", 1);
  pub_points = nh.advertise<robotx_msgs::PCL_points> ("/pcl_points", 1);
  ros::spin ();
}

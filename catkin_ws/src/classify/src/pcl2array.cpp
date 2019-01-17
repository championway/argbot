#include <ros/ros.h>
#include <cmath>
#include <sensor_msgs/PointCloud2.h>
#include "pcl_ros/point_cloud.h"
#include <pcl/io/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/io/pcd_io.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;
PointCloudXYZ::Ptr cloud (new PointCloudXYZ);
PointCloudXYZ::Ptr cloud_h (new PointCloudXYZ);
PointCloudXYZRGB::Ptr cloud_in (new PointCloudXYZRGB); 
sensor_msgs::PointCloud2 ros_out;
ros::Publisher pub_2d_pcl;
ros::Publisher pub_PoseArray;

void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
{
	pcl::fromROSMsg (*input, *cloud);
	//========== Project To Ground ==========
	// Create a set of planar coefficients with X=Y=0,Z=1
	/*pcl::ModelCoefficients::Ptr c (new pcl::ModelCoefficients ());
	c->values.resize (4);
	c->values[0] = 0;
	c->values[1] = 0;
	c->values[2] = 1.0;
	c->values[3] = 0;
	// Create the filtering object
	pcl::ProjectInliers<pcl::PointXYZ> proj;
	proj.setModelType (pcl::SACMODEL_PLANE);
	proj.setInputCloud (cloud);
	proj.setModelCoefficients (c);
	proj.filter (*cloud);*/

	pcl::ExtractIndices<pcl::PointXYZ> extract_h_l_place;
	pcl::PointIndices::Ptr hl_indices (new pcl::PointIndices);
	for (int i = 0; i < cloud->points.size(); i++)
	{
		if(cloud->points[i].y <=3.2 && cloud->points[i].y >= -3.5 && cloud->points[i].x >= -1.5 && cloud->points[i].x <= 1.5)
		{
			hl_indices->indices.push_back(i);
		}
	}
	extract_h_l_place.setInputCloud(cloud);
	extract_h_l_place.setIndices(hl_indices);
	extract_h_l_place.setNegative(true);
	extract_h_l_place.filter(*cloud_h);
	*cloud = *cloud_h;

	geometry_msgs::PoseArray pose_arr;
	for (size_t i = 0; i < cloud->points.size(); i++){
		geometry_msgs::Pose p;
		p.position.x = cloud->points[i].x;
		p.position.y = cloud->points[i].y;
		p.position.z = cloud->points[i].z;
		pose_arr.poses.push_back(p);
	}
	pub_PoseArray.publish(pose_arr);
	pub_2d_pcl.publish(*cloud);
}


int main (int argc, char** argv)
{
	// Initialize ROS
	ros::init (argc, argv, "pcl2array");
	ros::NodeHandle nh;
	std::cout<<"Start"<<std::endl;
	ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/velodyne_points", 1, cloud_cb);
	pub_2d_pcl = nh.advertise<PointCloudXYZ> ("/2d_pcl", 1);
	pub_PoseArray = nh.advertise<geometry_msgs::PoseArray> ("/pcl_array", 1);
	// Spin
	ros::spin ();
}
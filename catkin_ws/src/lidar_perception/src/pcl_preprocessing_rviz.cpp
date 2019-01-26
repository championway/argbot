/**********************************
Author: David Chen
Date: 2019/01/06
Last update: 2019/01/06
Point Cloud Preprocess
Subscribe: 
  /velodyne_points      			(sensor_msgs/PointCloud2)
Publish:
  /velodyne_points_preprocess		(sensor_msgs/PointCloud2)
  /velodyne_points_odom				(sensor_msgs/PointCloud2)
  /boundary_marker					(visualization_msgs/Marker)
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
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <geographic_msgs/GeoPoint.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
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
#include "tf/transform_datatypes.h"
#include <tf_conversions/tf_eigen.h>

#define PI 3.14159
using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;


class PreprocessNode{
private:
	string node_name;
	double range_min;
	double range_max;
	double angle_min;
	double angle_max;
	double z_max;
	double z_min;
	double y_min;
	double x_min;
	double z_min_near;

	vector< array<double, 2> >boundary_list;

	int counts;

	ros::NodeHandle nh;
	ros::Subscriber sub_cloud;
    ros::Subscriber sub_point;
	ros::Publisher	pub_cloud;
	ros::Publisher	pub_marker;
	ros::Publisher	pub_marker_point;

	//ros::ServiceClient client
	ros::ServiceServer service;
	//std_srvs::Empty srv;

	tf::TransformListener listener;

public:
	PreprocessNode(ros::NodeHandle&);
	void getBoundaryXY(vector<double>&, int);
	void cbCloud(const sensor_msgs::PointCloud2ConstPtr&);
	void cbPoint(const geometry_msgs::PoseWithCovarianceStampedPtr&);
	bool pointInBoundary(double, double);
	double product(double, double, double, double);
	void drawBoundary();
	bool clear_bounary(std_srvs::Trigger::Request&, std_srvs::Trigger::Response&);
	void sortBoundary();
};

PreprocessNode::PreprocessNode(ros::NodeHandle &n){
	nh = n;
	counts = 0;
	node_name = ros::this_node::getName();

	range_min = 0.0;
	range_max = 30.0;
	angle_min = 0.0;
	angle_max = 180.0;
	z_min = -1.7;
	z_max = 5.0;
	y_min = 3.5;
	x_min = 1.5;
	z_min_near = -1.4;

	//Read yaml file
	nh.getParam("range_min", range_min);
	nh.getParam("range_max", range_max);
	nh.getParam("angle_min", angle_min);
	nh.getParam("angle_max", angle_max);
	nh.getParam("z_min", z_min);
	nh.getParam("z_max", z_max);
	nh.getParam("y_min", y_min);
	nh.getParam("x_min", x_min);
	nh.getParam("z_min_near", z_min_near);

	//client = nh.serviceClient<std_srvs::Empty>("/clear_pointcloud_boundary");
	service = nh.advertiseService("/clear_pointcloud_boundary", &PreprocessNode::clear_bounary, this);

	ROS_INFO("[%s] Initializing ", node_name.c_str());
	ROS_INFO("[%s] Param [range_max] = %f, [range_min] = %f", node_name.c_str(), range_max, range_min);
	ROS_INFO("[%s] Param [angle_max] = %f, [angle_min] = %f", node_name.c_str(), angle_max, angle_min);
	ROS_INFO("[%s] Param [x_min] = %f, [y_min] = %f", node_name.c_str(), x_min, y_min);
	ROS_INFO("[%s] Param [z_max] = %f, [z_min] = %f", node_name.c_str(), z_max, z_min);


	// Publisher
	pub_cloud = nh.advertise< sensor_msgs::PointCloud2 >("velodyne_points_preprocess", 1);
	pub_marker = nh.advertise< visualization_msgs::Marker >("boundary_marker", 1);
	pub_marker_point = nh.advertise< visualization_msgs::Marker >("boundary_marker_point", 1);

	// Subscriber
	sub_cloud = nh.subscribe("velodyne_points", 1, &PreprocessNode::cbCloud, this);
    sub_point = nh.subscribe("/initialpose", 1, &PreprocessNode::cbPoint, this);

}

void PreprocessNode::drawBoundary(){
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
	points.header.frame_id = "/odom";
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
	line_list.header.frame_id = "/odom";
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

bool PreprocessNode::clear_bounary(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res){
	res.success = 1;
	res.message = "Clear all boundary points";
	cout << "Clear all boundary points" << endl;
	boundary_list.clear();
	boundary_list.shrink_to_fit();
	return true;
}

bool PreprocessNode::pointInBoundary(double x, double y){
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

double PreprocessNode::product(double v1_x, double v1_y, double v2_x, double v2_y){
	return (v1_x*v2_y - v2_x*v1_y);
}

void PreprocessNode::cbPoint(const geometry_msgs::PoseWithCovarianceStampedPtr& point_msg){
    float x = point_msg->pose.pose.position.x;
    float y = point_msg->pose.pose.position.y;
	array<double, 2> point= {x, y};
	boundary_list.push_back(point);
	sortBoundary();
	
	// Draw the boudary area
	drawBoundary();
}

void PreprocessNode::sortBoundary(){
	//cout << " input = "<< point.at(0) <<", " << point.at(1) << ", angle = " << point_angle << endl ;
	double center_x = 0;
	double center_y = 0;
	for (int i=0; i< boundary_list.size() ;i++){
		array<double, 2>base = boundary_list[i];
		center_x += base.at(0);
		center_y += base.at(1);
	}
	center_x = center_x / (boundary_list.size());
	center_y = center_y / (boundary_list.size());
	//cout << center_x << "  " << center_y << endl;
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
		//cout << "" << boundary_list[index].at(0) <<", " << boundary_list[index].at(1) << ", angle = " << max_angle << endl;
	}
	//cout << "============================" << endl;
	boundary_list = sort_boundary;
	delete already_find;
	/*
	for (int i=0; i< boundary_list.size() ;i++){
		array<double, 2>base = boundary_list[i];
		cout << base.at(0) << "  " << base.at(1) << endl;
	}	
	cout << "---------------"<<endl;
	*/
		
}

void PreprocessNode::cbCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg){
	
	counts++ ;
	//return if no cloud data
	if ((cloud_msg->width * cloud_msg->height) == 0 || counts % 3 == 0)
		return ;
	const clock_t t_start = clock();
	
	// transfer ros msg to point cloud
	PointCloudXYZ::Ptr cloud(new PointCloudXYZ);
	PointCloudXYZ::Ptr cloud_far(new PointCloudXYZ);
	PointCloudXYZ::Ptr cloud_near(new PointCloudXYZ);
	PointCloudXYZ::Ptr cloud_tmp(new PointCloudXYZ);
	PointCloudXYZ::Ptr cloud_odom(new PointCloudXYZ);
	
	pcl::fromROSMsg(*cloud_msg, *cloud_tmp);
	pcl::fromROSMsg(*cloud_msg, *cloud_odom);

	string source_frame="/odom";
	string target_frame="/velodyne";
	tf::StampedTransform transformStamped;
	Eigen::Affine3d transform_eigen;
	try{
		listener.waitForTransform(source_frame, target_frame, ros::Time(), ros::Duration(2.0) );
		listener.lookupTransform(source_frame, target_frame, ros::Time(), transformStamped);
		
		tf::transformTFToEigen(transformStamped, transform_eigen);
		pcl::transformPointCloud(*cloud_tmp, *cloud_odom, transform_eigen);
	} 	
	catch (tf::TransformException ex) {
		ROS_INFO("[%s] Can't find transfrom betwen [%s] and [%s] ", node_name.c_str(), source_frame.c_str(), target_frame.c_str());		
		return;
	}

	/*
	***************************************************************
		Remove out of range points and WAM-V points
		Remove our of boundary points
	***************************************************************
	*/
	float dis, angle = 0;
	int num_far, num_near, num = 0;

	for (int i=0 ; i <  cloud_tmp->points.size() ; i++)
	{
		dis = cloud_tmp->points[i].x * cloud_tmp->points[i].x +
				cloud_tmp->points[i].y * cloud_tmp->points[i].y;
		dis = sqrt(dis);
		angle = atan2f(cloud_tmp->points[i].y, cloud_tmp->points[i].x);
		angle = angle * 180 / 3.1415;
		if (dis >= range_min)
		{
			if (dis <= range_max && cloud_tmp->points[i].z >= z_min && z_max >= cloud_tmp->points[i].z) 
			{
				if(angle>=angle_min && angle<=angle_max && (abs(cloud_tmp->points[i].y) >= y_min or abs(cloud_tmp->points[i].x) >= x_min))
				{
					if(pointInBoundary(cloud_odom->points[i].x, cloud_odom->points[i].y) )
					{
						cloud_far->points.push_back(cloud_tmp->points[i]);
						num_far ++;					
					}
					else;
				}
			
			}
		}
		else
		{
			if (dis <= range_max && cloud_tmp->points[i].z >= z_min_near && z_max >= cloud_tmp->points[i].z) 
			{
				if(angle>=angle_min && angle<=angle_max && (abs(cloud_tmp->points[i].y) >= y_min or abs(cloud_tmp->points[i].x) >= x_min))
				{
					if(pointInBoundary(cloud_odom->points[i].x, cloud_odom->points[i].y) )
					{
						cloud_near->points.push_back(cloud_tmp->points[i]);
						num_near ++;					
					}
					else;
				}
			
			}
		}
	}
	/*cloud_far->width = num_far;
	cloud_far->height = 1;
	cloud_far->points.resize(num_far);
	cloud_near->width = num_near;
	cloud_near->height = 1;
	cloud_near->points.resize(num_near);*/
	for (int i=0 ; i <  cloud_near->points.size() ; i++)
	{
		cloud->points.push_back(cloud_near->points[i]);
		num ++;
	}
	for (int i=0 ; i <  cloud_far->points.size() ; i++)
	{
		cloud->points.push_back(cloud_far->points[i]);
		num ++;
	}
	cloud->width = num;
	cloud->height = 1;
	cloud->points.resize(num);
	//cout << cloud_far->points.size()+cloud_near->points.size() << ',' << cloud->points.size() << endl;
	clock_t t_end = clock();
	//cout << "PointCloud preprocess time taken = " << (t_end-t_start)/(double)(CLOCKS_PER_SEC) << endl;

	sensor_msgs::PointCloud2 cloud_out;
	pcl::toROSMsg(*cloud, cloud_out);
	cloud_out.header = cloud_msg->header;
	pub_cloud.publish(cloud_out);

}

int main(int argc, char **argv){
	ros::init (argc, argv, "pcl_preprocessing_node");
	ros::NodeHandle nh("~");
	PreprocessNode pn(nh);
	
	ros::spin ();
	return 0;
}

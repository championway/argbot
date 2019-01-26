/*
Author: Tony Hsiao                                                              
Date: 2018/07/20                                                               
PlacardExtraction
        Input:  ObstaclePosCloud
        Output: Vector
*/
#include <cmath>
#include <cstdio>
#include <math.h>
#include <iostream>
//pcl library
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl_ros/point_cloud.h>
//ros library
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "geometry_msgs/Point.h" 
#include "geometry_msgs/Vector3.h" 
#include "std_msgs/Bool.h"
#include "robotx_msgs/ObjectPose.h"
#include "robotx_msgs/ObjectPoseList.h"
#include "robotx_msgs/PlacardPose.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;
class PlacardExtraction{
public:
    PlacardExtraction(ros::NodeHandle&);
    void cb_obj_list(const robotx_msgs::ObjectPoseListConstPtr&);
    void cal_normal_avg(pcl::PointCloud<pcl::PointXYZINormal> );
private:
    string node_name;
    bool visual;

    ros::NodeHandle nh;
    ros::Subscriber sub_ob_list;
    ros::Publisher pub_marker;
    ros::Publisher pub_pose;

};
PlacardExtraction::PlacardExtraction(ros::NodeHandle& n){
    nh = n;
    node_name = ros::this_node::getName();

    //Initial param
    visual = nh.param("visual", true);

	ROS_INFO("[%s] Initializing ", node_name.c_str());
	ROS_INFO("[%s] Param [visual] = %d", node_name.c_str(), visual);

    //Publisher
    pub_marker = nh.advertise<visualization_msgs::MarkerArray>("marker_cloud_normal", 100);
    pub_pose   = nh.advertise<robotx_msgs::PlacardPose>("dock_pose", 1);

    //Subscriber
    sub_ob_list = nh.subscribe("/obj_list/odom", 1, &PlacardExtraction::cb_obj_list, this);

}
void PlacardExtraction::cb_obj_list(const robotx_msgs::ObjectPoseListConstPtr& msg_obj_list){
    robotx_msgs::ObjectPose dock; 
    dock.type = "nothing";   

    for (int i = 0 ; i < msg_obj_list->size ; i++){
        if (msg_obj_list->list[i].type == "dock"){
            dock = msg_obj_list->list[i];
        }
    }

	//return if no cloud data
	if (dock.type == "nothing" || (dock.cloud.width * dock.cloud.height) == 0)
		return ;    

    const clock_t t_start = clock();

 	// transfer ros msg to point cloud
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromROSMsg(dock.cloud, *cloud_tmp);
    int num = 0;
    // Filter the points under z threshold
    for (int i = 0 ; i < cloud_tmp->points.size() ; i++){
        if(cloud_tmp->points[i].z >= -2){
            cloud->points.push_back(cloud_tmp->points[i]);
            num ++;
        }
    }    
	cloud->width = num ;
	cloud->height = 1;
	cloud->points.resize(num);
   
	/*
	***************************************************************
		Find average Normal Estimation
	***************************************************************
	*/
    // Find Normal Estimation
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normal(new pcl::PointCloud<pcl::PointXYZINormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
   
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    // Use all neighbors in a sphere of radius
    ne.setRadiusSearch (0.7);
    ne.compute (*cloud_normals);

    pcl::concatenateFields( *cloud, *cloud_normals, *cloud_with_normal );

    cal_normal_avg(*cloud_with_normal);

    const clock_t t_end = clock();
    //cout << "Normal Estimation time taken = " << (t_end - t_start)/(double)(CLOCKS_PER_SEC)  << endl;

}
void PlacardExtraction::cal_normal_avg(pcl::PointCloud<pcl::PointXYZINormal> cloud_with_normals){
	/*
	***************************************************************
		Visualize normal
	***************************************************************
	*/
    visualization_msgs::Marker marker_1;
    visualization_msgs::MarkerArray marker_array;
    marker_1.header.stamp = ros::Time::now();
    marker_1.header.frame_id = "velodyne";
    marker_1.id = 2;
    
    marker_1.type = visualization_msgs::Marker::LINE_LIST;
    marker_1.action = visualization_msgs::Marker::ADD;

    marker_1.color.r = 1.0f;
    marker_1.color.g = 1.0f;
    marker_1.color.b = 0.0f;
    marker_1.color.a = 1.0;

    marker_1.scale.x = 0.01;
    marker_1.scale.y = 0.01;
    marker_1.lifetime = ros::Duration();

    visualization_msgs::Marker marker_2(marker_1);

    int counts = 0;
    double sum_x = 0, sum_y = 0, sum_z = 0, sum_normal_x = 0, sum_normal_y = 0, sum_normal_z = 0;
    for(int i=0;i<cloud_with_normals.points.size();i++){
        if ( !isnan(cloud_with_normals.points[i].normal_x) && abs(cloud_with_normals.points[i].normal_z) < 0.5){
            sum_x += cloud_with_normals.points[i].x;
            sum_y += cloud_with_normals.points[i].y;
            sum_z += cloud_with_normals.points[i].z;
            sum_normal_x += cloud_with_normals.points[i].normal_x;
            sum_normal_y += cloud_with_normals.points[i].normal_y;
            sum_normal_z += cloud_with_normals.points[i].normal_z;
            counts ++;

            //Visualize all normals
            if (visual){
                
                geometry_msgs::Point p_st, p_ed;
                p_st.x = cloud_with_normals.points[i].x;
                p_st.y = cloud_with_normals.points[i].y;
                p_st.z = cloud_with_normals.points[i].z;
                marker_1.points.push_back(p_st);

                p_ed.x = p_st.x + cloud_with_normals.points[i].normal_x * 3;
                p_ed.y = p_st.y + cloud_with_normals.points[i].normal_y * 3; 
                p_ed.z = p_st.z + cloud_with_normals.points[i].normal_z * 3;     
                marker_1.points.push_back(p_ed); 
                

                //cout << cloud_with_normals.points[i].normal_x << " " << cloud_with_normals.points[i].normal_y << " " 
                //    << cloud_with_normals.points[i].normal_z << " " << cloud_with_normals.points[i].curvature << endl;  
            }

            
        }
    }
    marker_array.markers.push_back(marker_1);
    if(counts !=0){
        double avg_x, avg_y, avg_z, avg_normal_x, avg_normal_y, avg_normal_z;
        avg_x = sum_x / counts;
        avg_y = sum_y / counts;
        avg_z = sum_z / counts;
        avg_normal_x = sum_normal_x / counts;
        avg_normal_y = sum_normal_y / counts;
        avg_normal_z = sum_normal_z / counts;
        if (visual){

            marker_2.color.r = 0.0f;
            marker_2.color.g = 0.0f;
            marker_2.color.b = 1.0f;
            marker_2.scale.x = 0.5;
            marker_2.scale.y = 0.5;
            marker_2.id = 3;

            geometry_msgs::Point p_st, p_ed;
            p_st.x = avg_x;
            p_st.y = avg_y;
            p_st.z = avg_z;
            marker_2.points.push_back(p_st);

            p_ed.x = p_st.x + avg_normal_x * 3;
            p_ed.y = p_st.y + avg_normal_y * 3; 
            p_ed.z = p_st.z + avg_normal_z * 3;     
            marker_2.points.push_back(p_ed);

            marker_array.markers.push_back(marker_2);
            pub_marker.publish(marker_array);
        }
        //pub normal direction of pose
        robotx_msgs::PlacardPose pose;
        pose.position.x = avg_x;
        pose.position.y = avg_y;
        pose.position.z = avg_z;        
        pose.normal.x = avg_normal_x;
        pose.normal.y = avg_normal_y;
        pose.normal.z = avg_normal_z;
        pub_pose.publish(pose);
    }
    
        
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "placard_extraction");
    ros::NodeHandle nh("~");
    PlacardExtraction pe(nh);
    
    ros::spin();
    return 0;
}

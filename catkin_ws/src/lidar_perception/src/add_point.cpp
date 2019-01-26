/**********************************
Author: David Chen
Date: 2018/06/01 
Last update: 2018/08/22                                                              
Point Cloud Clustering
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
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <Eigen/Dense>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <gazebo_msgs/SetModelState.h>
#include <gazebo_msgs/GetModelState.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pcl/io/ply_io.h>
//TF lib
#include <tf/transform_listener.h>
#include "tf/transform_datatypes.h"
#include <tf_conversions/tf_eigen.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace Eigen;
using namespace message_filters;
//define point cloud type

typedef uint32_t uint32;
struct PointCloudLabel
{
  PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
  PCL_ADD_RGB;
  uint32 label;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointCloudLabel,           // here we assume a XYZ + "test" (as fields)
                                   (float, x, x)
                                   (float, y, y)
                                   (float, z, z)
                                   (float, rgb, rgb)
                                   (uint32, label, label)
)

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudXYZRGB;
typedef pcl::PointCloud<PointCloudLabel> PointLabel;

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
PointLabel::Ptr label_cloud (new PointLabel);

sensor_msgs::PointCloud2 ros_out;
sensor_msgs::PointCloud2 ros_cluster;

//declare ROS publisher
ros::Publisher pub_result;
ros::Publisher pub_marker;
ros::Publisher pub_marker_line;
ros::Publisher pub_obstacle;
ros::Publisher pub_object;
ros::Publisher pub_points;
ros::ServiceClient client;
ros::ServiceClient get_client;

//declare global variable
std_msgs::String pcl_frame_id; 
bool lock = false;
float low = -0.25;
float high = 1.5-low;
float thres_low = 0.03;
float thres_high = 1.5;
int counter = 0;
bool visual;
float feature_sampling_space = 0.1;
visualization_msgs::MarkerArray marker_array;
visualization_msgs::MarkerArray marker_array_line;
ros::Time pcl_t;

int MODEL_NUM = 1;
std::string model_id[1] = {"dock"};
int class_dic[1] = {3};
float z_height[1] = {0};
float new_pos_arr[1][2];
float new_pos_arr_tf[1][2];
float pos_arr[1][2];
float pos_arr_tf[1][2];
bool first = true;
int pcd_count = 186;

//declare function
void cloud_cb(const sensor_msgs::PointCloud2ConstPtr&); //point cloud subscriber call back function
void cluster_pointcloud(void); //point cloud clustering
void set_gazebo_world(std::string model_name, int x, int y, float z);
void get_model_state(std::string model_name);
void auto_label(void);
void drawRviz(robotx_msgs::ObstaclePoseList); //draw marker in Rviz
void drawRviz_line(robotx_msgs::ObstaclePoseList); //draw marker line list in Rviz

tf::TransformListener* lr;
int gazebo_world_counter = 0;
int move_x = 0;
tf::StampedTransform tf_transform;

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
      if (cloud_in->points[i].y > 0)
      {
      	cloud_in->points[i].r = 255;
      	cloud_in->points[i].g = 0;
      	cloud_in->points[i].b = 255;
      }
      else
      {
      	cloud_in->points[i].r = 0;
      	cloud_in->points[i].g = 255;
      	cloud_in->points[i].b = 0;
      }
      
    }
    clock_t t_start = clock();

    std::string source_frame="/velodyne";
	std::string target_frame="/base_link";
	try{
		lr->lookupTransform(source_frame, target_frame, ros::Time(), tf_transform);
	} 	
	catch (tf::TransformException ex) {
		ROS_INFO("Can't find transfrom betwen [%s] and [%s] ", source_frame.c_str(), target_frame.c_str());		
		return;
	}
    cluster_pointcloud();
    clock_t t_end = clock();
    //std::cout << "Pointcloud cluster time taken = " << (t_end-t_start)/(double)(CLOCKS_PER_SEC) << std::endl;
  }
  else{
    std::cout << "lock" << std::endl;
  }

}

void get_model_state(std::string model_name)
{
  gazebo_msgs::ModelState modelstate;
  //modelstate.model_name = (std::string) model_name;

  gazebo_msgs::GetModelState getmodelstate;
  getmodelstate.request.model_name = model_name;
  get_client.call(getmodelstate);
}

void set_gazebo_world(std::string model_name, int x, int y, float z)
{
	gazebo_msgs::ModelState modelstate;
    modelstate.model_name = (std::string) model_name;
    modelstate.pose.position.x = x;
    modelstate.pose.position.y = y;
    modelstate.pose.position.z = z;
    /*if (model_name == "wamv")
    {
    	tf2::Quaternion quat;
    	quat.setRPY(0, 0, 3.14);
    	modelstate.pose.orientation.x = quat[0];
    	modelstate.pose.orientation.y = quat[1];
    	modelstate.pose.orientation.z = quat[2];
    	modelstate.pose.orientation.w = quat[3];
    }*/
    
    gazebo_msgs::SetModelState setmodelstate;
    setmodelstate.request.model_state = modelstate;
    client.call(setmodelstate);
}

void auto_label()
{
	int RAND_RANGE = 35;
	float COLLISION_DIS = 5;
	float WAMV_RANGE = 8;
	srand(time(NULL));
	for(int i = 0; i < MODEL_NUM; i++)
	{
		bool model_collision = true;
		//To get both positive and negetive value
		int x = (int)(rand()%RAND_RANGE*2 + 1) - RAND_RANGE;
		int y = (int)(rand()%RAND_RANGE*2 + 1) - RAND_RANGE;
		new_pos_arr[i][0] = x;
		new_pos_arr[i][1] = y;
		while(model_collision)
		{
			model_collision = false;
			x = (int)(rand()%RAND_RANGE*2 + 1) - RAND_RANGE;
			y = (int)(rand()%RAND_RANGE*2 + 1) - RAND_RANGE;
			new_pos_arr[i][0] = x;
			new_pos_arr[i][1] = y;

			if (sqrt(pow(x, 2) + pow(y, 2)) < WAMV_RANGE){model_collision = true;};
			float dis;
			for(int j = 0; j < i; j++)
			{
				dis = sqrt(pow((new_pos_arr[j][0]-new_pos_arr[i][0]), 2) + pow((new_pos_arr[j][1]-new_pos_arr[i][1]), 2));
				if (dis < COLLISION_DIS){model_collision = true;}
			}
		}
		set_gazebo_world(model_id[i], x, y, z_height[i]);
		tf::Quaternion quat = tf_transform.getRotation();
  		tf::Matrix3x3 tf_rot(quat);
  		tf::Vector3 pos(x, y, 0);
  		tf::Vector3 tf_pos = tf_rot * pos;
  		new_pos_arr_tf[i][0] = tf_pos[0];
  		new_pos_arr_tf[i][1] = tf_pos[1];
	}
	set_gazebo_world("wamv", 0, 0, -0.0823);
}

//void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input)
void cluster_pointcloud()
{
  //std::cout<< "start processing point clouds" << std::endl;
  copyPointCloud(*cloud_in, *cloud_filtered);  

  //========== Outlier remove ==========
  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
  // build the filter
  outrem.setInputCloud(cloud_filtered);
  outrem.setRadiusSearch(6);
  outrem.setMinNeighborsInRadius (2);
  // apply filter
  outrem.filter (*cloud_filtered);

  //========== Auto Labeling ==========
  for (int i = 0; i < MODEL_NUM; i++)
  {
  	for (int j = 0; j < 2; j++)
  	{
  		pos_arr_tf[i][j] = new_pos_arr_tf[i][j];
  	}
  }
  auto_label();

  ros::Duration(3).sleep();
  if (first)
  {
  	first = false;
  	lock = false;
  	return;
  }

  label_cloud->points.resize(cloud_filtered->points.size());
  label_cloud->width = cloud_filtered->width;
  label_cloud->height = cloud_filtered->height;
  
  for(size_t idx = 0; idx < cloud_filtered->points.size(); ++idx)
  {
  	float x = cloud_filtered->points[idx].x;
  	float y = cloud_filtered->points[idx].y;
  	float z = cloud_filtered->points[idx].z;
  	float min_dis = 100000000;
  	int min_idx;
  	for(int i = 0; i < MODEL_NUM; i++)
  	{
  		
  		float dis = sqrt(pow((x-pos_arr_tf[i][0]), 2) + pow((y-pos_arr_tf[i][1]), 2));
  		//float dis = sqrt(pow((x+pos_arr[i][1]), 2) + pow((y-pos_arr[i][0]), 2));
  		if(min_dis > dis)
  		{
  			min_dis = dis;
  			min_idx = i;
  		}
  	}
  	
  	label_cloud->points[idx].x = x;
  	label_cloud->points[idx].y = y;
  	label_cloud->points[idx].z = z;
  	label_cloud->points[idx].label = class_dic[min_idx];
  	if (label_cloud->points[idx].label == 1)
  	{
  		label_cloud->points[idx].r = 0;
  		label_cloud->points[idx].g = 255;
  		label_cloud->points[idx].b = 0;
  	}
  	else if (label_cloud->points[idx].label == 2)
  	{
  		label_cloud->points[idx].r = 255;
  		label_cloud->points[idx].g = 255;
  		label_cloud->points[idx].b = 0;
  	}
  	else if (label_cloud->points[idx].label == 3)
  	{
  		label_cloud->points[idx].r = 255;
  		label_cloud->points[idx].g = 255;
  		label_cloud->points[idx].b = 255;
  	}
  	else if (label_cloud->points[idx].label == 4)
  	{
  		label_cloud->points[idx].r = 0;
  		label_cloud->points[idx].g = 0;
  		label_cloud->points[idx].b = 255;
  	}
  	else if (label_cloud->points[idx].label == 5)
  	{
  		label_cloud->points[idx].r = 0;
  		label_cloud->points[idx].g = 0;
  		label_cloud->points[idx].b = 255;
  	}
  	else if (label_cloud->points[idx].label == 6)
  	{
  		label_cloud->points[idx].r = 0;
  		label_cloud->points[idx].g = 0;
  		label_cloud->points[idx].b = 255;
  	}
  }

  



  
  //========== Point Cloud Clusterlabeling ==========
  // Declare variable
  int num_cluster = 0;

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
  tree->setInputCloud (cloud_filtered);

  // Create cluster object
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
  ec.setClusterTolerance (3.7);// unit: meter
  ec.setMinClusterSize (5);
  ec.setMaxClusterSize (100000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    Eigen::Vector4f centroid;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);
    PointLabel::Ptr cloud_label (new PointLabel);

    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    {
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
      result->points.push_back(cloud_filtered->points[*pit]);
    }
    int n_times = ceil(2048./cloud_cluster->points.size());
    int pcd_size = cloud_cluster->points.size() + cloud_cluster->points.size()*n_times;
    cloud_label->points.resize(pcd_size);
    cloud_label->width = pcd_size;
    cloud_label->height = 1;
    int pcd_idx = 0;
    for (size_t i = 0; i < cloud_cluster->points.size(); i++){
        cloud_label->points[pcd_idx].x = cloud_cluster->points[i].x;
        cloud_label->points[pcd_idx].y = cloud_cluster->points[i].y;
        cloud_label->points[pcd_idx].z = cloud_cluster->points[i].z;
        cloud_label->points[pcd_idx].r = 0;
        cloud_label->points[pcd_idx].g = 255;
        cloud_label->points[pcd_idx].b = 0;
        cloud_label->points[pcd_idx].label = 5;
        pcd_idx++;
        for (int j = 0; j < n_times; j++)
        {
          float dx = (rand()%601)/10000. - 0.03;
          float dy = (rand()%601)/10000. - 0.03;
          float dz = (rand()%601)/10000. - 0.03;
          cloud_label->points[pcd_idx].x = cloud_cluster->points[i].x + dx;
          cloud_label->points[pcd_idx].y = cloud_cluster->points[i].y + dy;
          cloud_label->points[pcd_idx].z = cloud_cluster->points[i].z + dz;
          cloud_label->points[pcd_idx].r = 0;
          cloud_label->points[pcd_idx].g = 255;
          cloud_label->points[pcd_idx].b = 0;
          cloud_label->points[pcd_idx].label = 5;
          pcd_idx++;
        }
    }
    num_cluster++;
    std::ostringstream oss_pcd, oss_ply;
    oss_pcd << "/media/arg_ws3/5E703E3A703E18EB/pcd/dock/gazebo_dock_" << pcd_count << ".pcd";
    oss_ply << "/media/arg_ws3/5E703E3A703E18EB/pcd/dock/gazebo_dock_" << pcd_count << ".ply";
    std::string file_name_pcd = oss_pcd.str();
    std::string file_name_ply = oss_ply.str();
    if (cloud_label->points.size()!=0)
    {
      pcl::io::savePCDFile(file_name_pcd, *cloud_label);
      std::cout << "Save PDC file: " << pcd_count << ".pcd" << std::endl;
      pcl::io::savePLYFile (file_name_ply, *cloud_label);
      std::cout << "Save PLY file: " << pcd_count << ".ply" << std::endl;
      pcd_count++;
      cloud_label->clear();
    }
    //pcl::compute3DCentroid(*cloud_cluster, centroid);
    pcl::toROSMsg(*cloud_cluster, ros_cluster);
  }

  if (pcd_count > 400)
  {
    std::cout << "Finish Auto Labeling" << std::endl;
    ros::shutdown();
  }

  result->header.frame_id = cloud_in->header.frame_id;
  pcl::toROSMsg(*result, ros_out);
  ros_out.header.stamp = pcl_t;
  //ros_out.header.stamp = ros::Time::now();
  pub_result.publish(ros_out);
  result->clear();

  lock = false;

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
  ros::init (argc, argv, "auto_label");
  ros::NodeHandle nh("~");
  visual = nh.param("visual", true);
  ROS_INFO("[pcl_cluster] Param [visual] = %d",  visual);
  client = nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
  get_client = nh.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
  tf::TransformListener listener(ros::Duration(1.0));
  lr = &listener;
  if (visual)
    std::cout<< "Start to clustering" << std::endl;
  ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/pcl_preprocessing/velodyne_points_preprocess", 1, callback);
  // Create a ROS publisher for the output point cloud
  pub_obstacle = nh.advertise< robotx_msgs::ObstaclePoseList > ("/obstacle_list", 1);
  pub_object = nh.advertise< robotx_msgs::ObjectPoseList > ("/obj_list", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("/obstacle_marker", 1);
  pub_marker_line = nh.advertise<visualization_msgs::MarkerArray>("/obstacle_marker_line", 1);
  pub_result = nh.advertise<sensor_msgs::PointCloud2> ("/cluster_result", 1);
  pub_points = nh.advertise<robotx_msgs::PCL_points> ("/pcl_points", 1);
  ros::spin ();
}
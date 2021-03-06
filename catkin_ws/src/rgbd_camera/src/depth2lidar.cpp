#include <ros/ros.h>
#include <vector>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "realsense2_camera/Extrinsics.h"
using namespace ros;
using namespace std;
using namespace cv;
using namespace pcl;
using namespace message_filters;

bool is_gazebo = true;


class DEPTH2LIDAR{
  public:
    DEPTH2LIDAR();
    //void generate_pointcloud(PointCloud<PointXYZRGB>::Ptr);
    void get_msg();
    void callback(const sensor_msgs::ImageConstPtr&);
    void callback_sync(const sensor_msgs::ImageConstPtr&,const sensor_msgs::ImageConstPtr&);
    void getXYZ(float* , float* ,float );
  private:
  	Publisher pc2;
  	ros::Subscriber depth_image;
    PointCloud<PointXYZRGB>::Ptr pc;
    message_filters::Subscriber<sensor_msgs::Image> img_sub;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;
  	float fx;
  	float fy;
  	float cx;
  	float cy;
  	float Projection[3][4];
  	float extrinsics[3][4];
};

DEPTH2LIDAR::DEPTH2LIDAR(){
	NodeHandle nh;
	pc2 = nh.advertise<sensor_msgs::PointCloud2> ("/pc", 10);
	depth_image = nh.subscribe<sensor_msgs::Image>("/generate_dp", 1, &DEPTH2LIDAR::callback, this);
}

void DEPTH2LIDAR::getXYZ(float* a, float* b,float zc){

	float inv_fx = 1.0/fx;
	float inv_fy = 1.0/fy;
	*a = (*a - cx) * zc * inv_fx;
	*b = (*b - cy) * zc * inv_fy;
	return;
}
void DEPTH2LIDAR::callback(const sensor_msgs::ImageConstPtr& depth_image){
  pc.reset(new PointCloud<PointXYZRGB>());
  cv_bridge::CvImagePtr img_ptr_depth;
  if(is_gazebo){
    img_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16SC1);
  }
	else{
    img_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
  }
	for( int nrow = 10; nrow < img_ptr_depth->image.rows-10; nrow++){  
       for(int ncol = 10; ncol < img_ptr_depth->image.cols-10; ncol++){  
       	if (img_ptr_depth->image.at<unsigned short int>(nrow,ncol) > 1){
       		
       		pcl::PointXYZRGB point;
       		float* x = new float(nrow);
       		float* y = new float(ncol);
          float z;
          if(is_gazebo){
            z = float(img_ptr_depth->image.at<unsigned short int>(nrow,ncol))/1000.;
          }
       	 	else{
            z = float(img_ptr_depth->image.at<unsigned short int>(nrow,ncol))/1000.;
          }

       		getXYZ(y,x,z);
       		point.x = z;
       		point.y = -*y;
       		point.z = -*x;
       		point.r = int(255);
       		point.g = int(0);
       		point.b = int(255);
       		pc->points.push_back(point);
       		free(x);
       		free(y);
       	} 
       }  
    } 
    sensor_msgs::PointCloud2 object_cloud_msg;
    toROSMsg(*pc, object_cloud_msg);
    if(is_gazebo){
      object_cloud_msg.header.frame_id = "X1/rgbd_camera_link";
    }
    else{
      object_cloud_msg.header.frame_id = "camera_color_optical_frame";
    }
    pc2.publish(object_cloud_msg);
	return;
}

/*void DEPTH2LIDAR::callback_sync(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& depth_image){
  pc.reset(new PointCloud<PointXYZRGB>());
  cv_bridge::CvImagePtr img_ptr_depth;
  if(is_gazebo){
    img_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_32FC1);
  }
	else{
    img_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
  }
	cv_bridge::CvImagePtr img_ptr_img = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);
	for( int nrow = 0; nrow < img_ptr_depth->image.rows; nrow++){  
       for(int ncol = 0; ncol < img_ptr_depth->image.cols; ncol++){  
       	if (img_ptr_depth->image.at<unsigned short int>(nrow,ncol) > 1){
       		
       		pcl::PointXYZRGB point;
       		float* x = new float(nrow);
       		float* y = new float(ncol);
          float z;
          if(is_gazebo){
            z = float(img_ptr_depth->image.at<float>(nrow,ncol));
          }
       	 	else{
            z = float(img_ptr_depth->image.at<unsigned short int>(nrow,ncol))/1000.;
          }

       		getXYZ(y,x,z);
       		point.x = z;
       		point.y = -*y;
       		point.z = -*x;
       		Vec3b intensity =  img_ptr_img->image.at<Vec3b>(nrow, ncol); 
       		point.r = int(intensity[0]);
       		point.g = int(intensity[1]);
       		point.b = int(intensity[2]);
       		pc->points.push_back(point);
       		free(x);
       		free(y);
       		// delete x;
       		// delete y;
       	} 
       }  
    } 

    //cout << pc->points.size() << endl;
    
    sensor_msgs::PointCloud2 object_cloud_msg;
    toROSMsg(*pc, object_cloud_msg);
    if(is_gazebo){
      object_cloud_msg.header.frame_id = "X1/rgbd_camera_link";
    }
    else{
      object_cloud_msg.header.frame_id = "camera_color_optical_frame";
    }
    pc2.publish(object_cloud_msg);
	return;
}*/

void DEPTH2LIDAR::get_msg(){
  sensor_msgs::CameraInfo::ConstPtr msg;
  if(is_gazebo){
    msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/X1/rgbd_camera/rgb/camera_info",ros::Duration(10));
  }
  else{
    msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/color/camera_info",ros::Duration(10));
  }
	fx = msg->P[0];
	fy = msg->P[5];
	cx = msg->P[2];
	cy = msg->P[6];
  cout << fx << "," << fy << "," << cx << "," << cy << std::endl;
	int count = 0;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 4; j++)
			Projection[i][j] = msg->P[count++];

  if(!is_gazebo){
    realsense2_camera::ExtrinsicsConstPtr msg1 = ros::topic::waitForMessage<realsense2_camera::Extrinsics>("/camera/extrinsics/depth_to_color",ros::Duration(10));
  	count = 0;
  	for(int i = 0; i < 3; i++)
  		for(int j = 0; j < 3; j++)
  			extrinsics[i][j] = msg1->rotation[count++];
  	for(int i = 0; i < 3 ; i++)
  		extrinsics[i][3] = msg1->translation[i];
  }
	return;
}
int main(int argc, char** argv){
    init(argc, argv, "DEPTH2LIDAR");
    DEPTH2LIDAR DEPTH2LIDAR;
    DEPTH2LIDAR.get_msg();
    spin();
    return 0;
}
#include "depth_to_point_whole.h"

void depth_to_point_whole::getXYZ(float* a, float* b,float zc){

	float inv_fx = 1.0/fx;
	float inv_fy = 1.0/fy;
	*a = (*a - cx) * zc * inv_fx;
	*b = (*b - cy) * zc * inv_fy;
	return;
}

void depth_to_point_whole::callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& depth_image){
	cv_bridge::CvImagePtr img_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
	cv_bridge::CvImagePtr img_ptr_img = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);

	pc->clear();


	for( int nrow = 0; nrow < img_ptr_depth->image.rows; nrow++){  
       for(int ncol = 0; ncol < img_ptr_depth->image.cols; ncol++){  
       	if (img_ptr_depth->image.at<unsigned short int>(nrow,ncol) > 1){
       		
       		pcl::PointXYZRGB point;
       		float* x = new float(nrow);
       		float* y = new float(ncol);
       	 	float z = float(img_ptr_depth->image.at<unsigned short int>(nrow,ncol))/1000.;

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
    object_cloud_msg.header.frame_id = "camera_link";
    pc_pub_bbox.publish(object_cloud_msg);
	return;
}
depth_to_point_whole::depth_to_point_whole(){
	NodeHandle nh;
	time = ros::Time::now().toSec();
	pc_pub_bbox = nh.advertise<sensor_msgs::PointCloud2> ("/whole_pc", 10);
	pc.reset(new PointCloud<PointXYZRGB>());
	img_sub.subscribe(nh, "/camera/color/image_raw", 1);
	depth_sub.subscribe(nh, "/camera/aligned_depth_to_color/image_raw", 1);
	sync_.reset(new Sync(MySyncPolicy(1), img_sub, depth_sub));
	sync_->registerCallback(boost::bind(&depth_to_point_whole::callback, this, _1, _2));
}
void depth_to_point_whole::get_msg(){
	sensor_msgs::CameraInfo::ConstPtr msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/color/camera_info",ros::Duration());
	fx = msg->P[0];
	fy = msg->P[5];
	cx = msg->P[2];
	cy = msg->P[6];
	int count = 0;
	for(int i = 0; i < 3; i++)
		for(int j = 0; j < 4; j++)
			Projection[i][j] = msg->P[count++];

	return;
}
int main(int argc, char** argv){
    init(argc, argv, "depth_to_point_whole");
    depth_to_point_whole depth_to_point_whole;
    depth_to_point_whole.get_msg();
    spin();
    return 0;
}
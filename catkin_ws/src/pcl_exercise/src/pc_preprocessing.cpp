#include "pc_preprocessing.h"

void pc_preprocessing::getXYZ(float* a, float* b,float zc){

	float inv_fx = 1.0/fx;
	float inv_fy = 1.0/fy;
	*a = (*a - cx) * zc * inv_fx;
	*b = (*b - cy) * zc * inv_fy;
	return;
}

void pc_preprocessing::callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& depth_image){  // sensor_msgs::PointCloud2 pc_msg
	cv_bridge::CvImagePtr img_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
	cv_bridge::CvImagePtr img_ptr_img = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::RGB8);

	pc->clear();
	pc_filter->clear();
	pc_filter_2->clear();


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

	//pcl::fromROSMsg (pc_msg, *pc);

	if (pc->points.size()!=0){
		
		condrem.setCondition (range_cond);
		condrem.setInputCloud (pc);
		condrem.setKeepOrganized(true);
		// apply filter
		condrem.filter (*pc_filter);

		// Optional // plane filter 
		seg.setOptimizeCoefficients (true);
		// Mandatory
		seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);   // pcl::SACMODEL_PLANE   SACMODEL_PERPENDICULAR_PLANE
		seg.setMethodType (pcl::SAC_RANSAC);
		seg.setMaxIterations (200);								// 200 
		seg.setDistanceThreshold (0.3);
		/////////////set parameter of plane (degree x_y plane)
		Eigen::Vector3f axis = Eigen::Vector3f(0.0,0.0,1.0);	 // y : 1.0
		seg.setAxis(axis);
		seg.setEpsAngle(  50.0f * (PI/180.0f) );
		seg.setInputCloud (pc_filter);
		seg.segment (*inliers, *coefficients);

		// remove outlier 
		pc->clear();
		extract.setInputCloud (pc_filter);
		extract.setIndices (inliers);
		extract.setNegative (false);
		extract.filter (*pc);

		// Mandatory
		seg.setDistanceThreshold (0.025);
		/////////////set parameter of plane (degree x_y plane)
		seg.setAxis(axis);
		seg.setEpsAngle(  50.0f * (PI/180.0f) );
		seg.setInputCloud (pc);
		seg.segment (*inliers, *coefficients);

		// remove outlier 
		extract.setInputCloud (pc);
		extract.setIndices (inliers);
		extract.setNegative (true);
		extract.filter (*pc_filter_2);

		pc_filter_2->points.resize(pc_filter_2->points.size());
		pc_filter_2->width = pc_filter_2->points.size();
		pc_filter_2->height = 1;

		sensor_msgs::PointCloud2 object_cloud_msg;
		toROSMsg(*pc_filter_2, object_cloud_msg);
		object_cloud_msg.header.frame_id = "camera_color_optical_frame";
		pc_pre.publish(object_cloud_msg);
	}
}
pc_preprocessing::pc_preprocessing(){
	NodeHandle nh;
	time = ros::Time::now().toSec();
	pc_pre = nh.advertise<sensor_msgs::PointCloud2> ("/pc_preprocessing", 10);
	//pc_sub = nh.subscribe<sensor_msgs::PointCloud2>("/camera/depth_registered/points", 1, &pc_preprocessing::callback,this); 
	range_cond .reset(new pcl::ConditionAnd<PointXYZRGB> ());
	coefficients.reset(new pcl::ModelCoefficients);
	inliers.reset(new pcl::PointIndices);
	range_cond->addComparison (pcl::FieldComparison<PointXYZRGB>::ConstPtr (new pcl::FieldComparison<PointXYZRGB> ("x", pcl::ComparisonOps::GT, 0.5)));  // z
	range_cond->addComparison (pcl::FieldComparison<PointXYZRGB>::ConstPtr (new pcl::FieldComparison<PointXYZRGB> ("z", pcl::ComparisonOps::LT, 3.0)));	 // z
	pc.reset(new PointCloud<PointXYZRGB>()); 
	pc_filter.reset(new PointCloud<PointXYZRGB>()); 
	pc_filter_2.reset(new PointCloud<PointXYZRGB>()); 


	img_sub.subscribe(nh, "/camera/color/image_rect_color", 1);
	depth_sub.subscribe(nh, "/camera/aligned_depth_to_color/image_raw", 1);
	sync_.reset(new Sync(MySyncPolicy(1), img_sub, depth_sub));
	sync_->registerCallback(boost::bind(&pc_preprocessing::callback, this, _1, _2));
}
void pc_preprocessing::get_msg(){
	sensor_msgs::CameraInfo::ConstPtr msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/color/camera_info",ros::Duration());
	fx = msg->P[0];
	fy = msg->P[5];
	cx = msg->P[2];
	cy = msg->P[6];
	return;
}
int main(int argc, char** argv){
    init(argc, argv, "pc_preprocessing");
    pc_preprocessing pc_preprocessing;
    pc_preprocessing.get_msg();
    spin();
    return 0;
}
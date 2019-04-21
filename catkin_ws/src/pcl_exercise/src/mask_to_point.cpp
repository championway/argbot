#include "mask_to_point.h"

void mask_to_point::getXYZ(float* a, float* b,float zc){

	float inv_fx = 1.0/fx;
	float inv_fy = 1.0/fy;
	*a = (*a - cx) * zc * inv_fx;
	*b = (*b - cy) * zc * inv_fy;
	return;
}

void mask_to_point::callback(const bb_pointnet::bb_input msg){
	pc.reset(new PointCloud<PointXYZRGB>());
	cv_bridge::CvImagePtr img_ptr_depth = cv_bridge::toCvCopy(msg.depth, sensor_msgs::image_encodings::TYPE_16UC1);
	cv_bridge::CvImagePtr img_ptr_img = cv_bridge::toCvCopy(msg.image, sensor_msgs::image_encodings::RGB8);
	cv_bridge::CvImagePtr img_ptr_mask = cv_bridge::toCvCopy(msg.mask, sensor_msgs::image_encodings::TYPE_8UC1);

	for( int nrow = 0; nrow < img_ptr_depth->image.rows; nrow++){  
        for(int ncol = 0; ncol < img_ptr_depth->image.cols; ncol++){  
       		if (img_ptr_depth->image.at<unsigned short int>(nrow,ncol) > 1){

       			//cout << img_ptr_depth->image.at<unsigned short int>(nrow,ncol) << endl;
	       		pcl::PointXYZRGB point;
	       		float* x = new float(nrow);
	       		float* y = new float(ncol);
	       	 	float z = float(img_ptr_depth->image.at<unsigned short int>(nrow,ncol))/1000.;

	       		getXYZ(y,x,z);
	       		point.x = z;
	       		point.y = -*y;
	       		point.z = -*x;    
	       		int i = img_ptr_mask->image.at<uint8_t>(nrow,ncol);   		
				if (i == 0){		
					Vec3b intensity =  img_ptr_img->image.at<Vec3b>(nrow, ncol); 
					point.r = int(intensity[0]);
					point.g = int(intensity[1]);
					point.b = int(intensity[2]);
				}
				else{
					point.r = colormap[i][0];
					point.g = colormap[i][1];
					point.b = colormap[i][2];
				}
				pc->points.push_back(point);
				free(x);
				free(y);
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
mask_to_point::mask_to_point(){
	NodeHandle nh;

	pc_pub_bbox = nh.advertise<sensor_msgs::PointCloud2> ("/mask_to_point_result", 10);
	ssd_result = nh.subscribe<bb_pointnet::bb_input>("/bbox", 1, &mask_to_point::callback,this); 

}
void mask_to_point::get_msg(){
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
    init(argc, argv, "mask_to_point");
    mask_to_point mask_to_point;
    mask_to_point.get_msg();
    spin();
    return 0;
}
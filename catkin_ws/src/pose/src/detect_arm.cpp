#include <ros/ros.h>
#include <vector>
#include <pose_msgs/HumanPoses.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <math.h>

using namespace message_filters;

class detect_arm{
	private:
		std::vector<std::vector<std::vector<int> > > humanPoses;
		visualization_msgs::Marker marker;
		visualization_msgs::MarkerArray marker_array;
		ros::NodeHandle _nh, _pnh;
		ros::Subscriber sub_pose;
		ros::Publisher pub_arm;
		ros::Publisher pub_image;
		ros::Publisher pub_arms;
		ros::Publisher pc2;
		cv_bridge::CvImagePtr img_ptr_depth;
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc;
		geometry_msgs::Point goal_point;

		//Camera information
		float fx, fy, cx, cy;

		void callback_sync(const pose_msgs::HumanPoses::ConstPtr&,const sensor_msgs::ImageConstPtr&);
		void get_camerainfo();
		void getXYZ(float*, float*, float);
		void drawRviz();
		float find_depth(int x, int y, int kernal_size);

		message_filters::Subscriber<pose_msgs::HumanPoses> pose_sub;
		message_filters::Subscriber<sensor_msgs::Image> depth_sub;
		typedef message_filters::sync_policies::ApproximateTime<pose_msgs::HumanPoses, sensor_msgs::Image> MySyncPolicy;
		typedef message_filters::Synchronizer<MySyncPolicy> Sync;
		boost::shared_ptr<Sync> sync_;

	public:
		detect_arm(ros::NodeHandle nh, ros::NodeHandle pnh):
			_nh(nh),
			_pnh(pnh)
			{
				ROS_INFO("Start detectin");
				get_camerainfo();
				pub_arm = _pnh.advertise<visualization_msgs::Marker>("/arm_marker", 1);
				pub_image = nh.advertise<sensor_msgs::Image> ("/img", 1);
				pub_arms = _pnh.advertise<visualization_msgs::MarkerArray>("/arms_marker", 1);
				pose_sub.subscribe(_pnh, "/human_pose", 1);
				depth_sub.subscribe(_pnh, "/camera/aligned_depth_to_color/image_raw", 1);
				pc2 = nh.advertise<sensor_msgs::PointCloud2> ("/pc", 10);
				sync_.reset(new Sync(MySyncPolicy(10), pose_sub, depth_sub));
				sync_->registerCallback(boost::bind(&detect_arm::callback_sync, this, _1, _2));
				
				// sub_pose = _pnh.subscribe("/human_pose", 1, &detect_arm::cbPose, this);
			}
		~detect_arm(){
			ros::shutdown();
		}
};

void detect_arm::callback_sync(const pose_msgs::HumanPoses::ConstPtr& pose_msg, const sensor_msgs::ImageConstPtr& depth_image){
	try{
		img_ptr_depth = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
	}
	catch (cv_bridge::Exception& e){
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	humanPoses.resize(pose_msg->size);
	// int arm_count = 0;
	for (int i = 0; i < pose_msg->size; i++){
		humanPoses[i].resize(pose_msg->pose_list[i].poses.size());
		for (int j = 0; j < pose_msg->pose_list[i].poses.size(); j++){
			humanPoses[i][j].resize(2);
			humanPoses[i][j][0] = int(pose_msg->pose_list[i].poses[j].position.x);
			humanPoses[i][j][1] = int(pose_msg->pose_list[i].poses[j].position.y);
			// if (humanPoses[i][j][0] != -1 && humanPoses[i][j][1] != -1){
			// 	arm_count++;
			// }
		}
	}

	detect_arm::drawRviz();
}

void detect_arm::get_camerainfo(){
	sensor_msgs::CameraInfo::ConstPtr msg;
	bool get_camera_info = false;
	while(!get_camera_info){
		msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/depth/camera_info",ros::Duration(20));
		get_camera_info = true;
	}
	fx = msg->P[0];
	fy = msg->P[5];
	cx = msg->P[2];
	cy = msg->P[6];
	std::cout << fx << "," << fy << "," << cx << "," << cy << std::endl;
	return;
}

void detect_arm::getXYZ(float* a, float* b,float zc){
	float inv_fx = 1.0/fx;
	float inv_fy = 1.0/fy;
	*a = (*a - cx) * zc * inv_fx;
	*b = (*b - cy) * zc * inv_fy;
	return;
}

float detect_arm::find_depth(int x, int y, int kernal_size){
	std::vector<float> all_depth;
	// create depth kernal
	for (int i = -(kernal_size-1)/2; i < (kernal_size-1)/2; i++){
		if (i < 0 || i >= 640) continue;
		for (int j = -(kernal_size-1)/2; j < (kernal_size-1)/2; j++){
			if (j < 0 || j >= 480) continue;
			all_depth.push_back(float(img_ptr_depth->image.at<unsigned short int>(x+i, y+j))/1000.);
		}
	}

	// get median depth information
	size_t size = all_depth.size();
	if (size == 0){
		return 0.;
	}
	else{
		sort(all_depth.begin(), all_depth.end());
		if (size % 2 == 0){
			return (all_depth[size / 2 - 1] + all_depth[size / 2]) / 2;
		}
		else{
			return all_depth[size / 2];
		}
	}
}

void detect_arm::drawRviz(){
	marker_array.markers.resize(humanPoses.size() * 2);
	for (int i = 0; i < humanPoses.size(); i++){

		marker_array.markers[i].header.frame_id = "/camera_link";
		marker_array.markers[i].header.stamp = ros::Time();
		marker_array.markers[i].id = 0;
		marker_array.markers[i].type = visualization_msgs::Marker::LINE_STRIP;
		marker_array.markers[i].action = visualization_msgs::Marker::ADD;
		marker_array.markers[i].scale.x = 0.03;
		marker_array.markers[i].scale.y = 0.03;
		marker_array.markers[i].scale.z = 0.03;
		marker_array.markers[i].color.a = 1;
		marker_array.markers[i].color.r = 0.0;
		marker_array.markers[i].color.g = 1.0;
		marker_array.markers[i].color.b = 0.0;

		marker_array.markers[humanPoses.size() + i].header.frame_id = "/camera_link";
		marker_array.markers[humanPoses.size() + i].header.stamp = ros::Time();
		marker_array.markers[humanPoses.size() + i].id = 1;
		marker_array.markers[humanPoses.size() + i].type = visualization_msgs::Marker::SPHERE;
		marker_array.markers[humanPoses.size() + i].action = visualization_msgs::Marker::ADD;
		marker_array.markers[humanPoses.size() + i].scale.x = 0.08;
		marker_array.markers[humanPoses.size() + i].scale.y = 0.08;
		marker_array.markers[humanPoses.size() + i].scale.z = 0.08;
		marker_array.markers[humanPoses.size() + i].color.a = 1;
		marker_array.markers[humanPoses.size() + i].color.r = 1;
		marker_array.markers[humanPoses.size() + i].color.g = 1;
		marker_array.markers[humanPoses.size() + i].color.b = 0;

		int idx1 = 3;
		int idx2 = 4;
		bool detect_1 = (humanPoses[i][idx1][0] != -1 && humanPoses[i][idx1][1] != -1);
		bool detect_2 = (humanPoses[i][idx2][0] != -1 && humanPoses[i][idx2][1] != -1);
		if (detect_1 && detect_2){
			float* x1 = new float(humanPoses[i][idx1][1]);
			float* y1 = new float(humanPoses[i][idx1][0]);
			// float z1 = float(img_ptr_depth->image.at<unsigned short int>(int(*x1), int(*y1)))/1000.;
			float z1 = find_depth(int(*x1), int(*y1), 6);
			getXYZ(y1, x1, z1);
			geometry_msgs::Point p1;
			p1.x = z1;
			p1.y = -*y1;
			p1.z = -*x1;
			float* x2 = new float(humanPoses[i][idx2][1]);
			float* y2 = new float(humanPoses[i][idx2][0]);
			// float z2 = float(img_ptr_depth->image.at<unsigned short int>(int(*x2), int(*y2)))/1000.;
			float z2 = find_depth(int(*x2), int(*y2), 6);
			getXYZ(y2, x2, z2);
			geometry_msgs::Point p2;
			p2.x = z2;
			p2.y = -*y2;
			p2.z = -*x2;

			float v_x = p2.x - p1.x;
			float v_y = p2.y - p1.y;
			float v_z = p2.z - p1.z;
			float norm = sqrt(pow(v_x, 2) + pow(v_y, 2) + pow(v_z, 2));
			float dis = 0.8;
			goal_point.x = p2.x + (v_x/norm) * dis;
			goal_point.y = p2.y + (v_y/norm) * dis;
			goal_point.z = p2.z + (v_z/norm) * dis;

			marker_array.markers[i].points.clear();
			marker_array.markers[i].points.push_back(p1);
			marker_array.markers[i].points.push_back(p2);
			marker_array.markers[humanPoses.size() + i].pose.position = goal_point;

			circle(img_ptr_depth->image, cv::Point(int(humanPoses[i][idx1][0]), int(humanPoses[i][idx1][1])), 10, cv::Scalar(25600), 3, 8, 0);
			circle(img_ptr_depth->image, cv::Point(int(humanPoses[i][idx2][0]), int(humanPoses[i][idx2][1])), 10, cv::Scalar(25600), -1, 8, 0);
		}
	}
	cv_bridge::CvImage cvIMG;
	cvIMG.encoding = sensor_msgs::image_encodings::TYPE_16UC1;
	cvIMG.image = img_ptr_depth->image;
	pub_image.publish(cvIMG.toImageMsg());
	pub_arm.publish(marker);
	pub_arms.publish(marker_array);
}

int main(int argc, char** argv){
	ros::init(argc, argv, "detect_arm");
	ros::NodeHandle nh, pnh("~");
	detect_arm foo(nh, pnh);
	while(ros::ok()) ros::spinOnce();
	return 0;
}
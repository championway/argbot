#include <ros/ros.h>
#include <vector>
#include <pose_msgs/HumanPoses.h>
#include <geometry_msgs/Pose.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

using namespace message_filters;

class detect_arm{
	private:
		std::vector<std::vector<std::vector<int> > > humanPoses;
		visualization_msgs::Marker marker;
		visualization_msgs::MarkerArray marker_array;
		ros::NodeHandle _nh, _pnh;
		ros::Subscriber sub_pose;
		ros::Publisher pub_arm;
		ros::Publisher pub_arms;
		cv_bridge::CvImagePtr img_ptr_depth;

		//Camera information
		float fx, fy, cx, cy;

		void cbPose(const pose_msgs::HumanPoses&);
		void callback_sync(const pose_msgs::HumanPoses::ConstPtr&,const sensor_msgs::ImageConstPtr&);
		void get_camerainfo();
		void getXYZ(float&, float&, float);
		void drawRviz();

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
				pub_arms = _pnh.advertise<visualization_msgs::MarkerArray>("/arms_marker", 1);
				pose_sub.subscribe(_pnh, "/human_pose", 1);
				depth_sub.subscribe(_pnh, "/camera/depth/image_rect_raw", 1);
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

void detect_arm::cbPose(const pose_msgs::HumanPoses &msg){
	//std::cout<< msg.size <<std::endl;
	humanPoses.resize(msg.size);
	// int arm_count = 0;
	for (int i = 0; i < msg.size; i++){
		humanPoses[i].resize(msg.pose_list[i].poses.size());
		for (int j = 0; j < msg.pose_list[i].poses.size(); j++){
			humanPoses[i][j].resize(2);
			humanPoses[i][j][0] = int(msg.pose_list[i].poses[j].position.x);
			humanPoses[i][j][1] = int(msg.pose_list[i].poses[j].position.y);
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
		msg = ros::topic::waitForMessage<sensor_msgs::CameraInfo>("/camera/color/camera_info",ros::Duration(20));
		get_camera_info = true;
	}
	fx = msg->P[0];
	fy = msg->P[5];
	cx = msg->P[2];
	cy = msg->P[6];
	std::cout << fx << "," << fy << "," << cx << "," << cy << std::endl;
	return;
}

void detect_arm::getXYZ(float &a, float &b, float zc){
	float inv_fx = 1.0/fx;
	float inv_fy = 1.0/fy;
	a = (a - cx) * zc * inv_fx;
	b = (b - cy) * zc * inv_fy;
	return;
}

void detect_arm::drawRviz(){
	marker.header.frame_id = "camera_link";
	marker.header.stamp = ros::Time();
	marker.id = 0;
	marker.type = visualization_msgs::Marker::SPHERE;
	marker.action = visualization_msgs::Marker::ADD;
	marker.pose.orientation.x = 0.0;
	marker.pose.orientation.y = 0.0;
	marker.pose.orientation.z = 0.0;
	marker.pose.orientation.w = 1.0;
	marker.scale.x = 0.1;
	marker.scale.y = 0.1;
	marker.scale.z = 0.1;
	marker.color.a = 1.0;
	marker.color.r = 0.0;
	marker.color.g = 1.0;
	marker.color.b = 0.0;
	for (int i = 0; i < humanPoses.size(); i++){
		if (humanPoses[i][6][0] != -1 && humanPoses[i][6][1] != -1){
			float x = humanPoses[i][6][0];
			float y = humanPoses[i][6][1];
			float z = float(img_ptr_depth->image.at<unsigned short int>(y, x))/1000.;;
			std::cout<<x<<','<<y<<','<<z<<std::endl;
			getXYZ(y, x, z);
			std::cout<<x<<','<<y<<','<<z<<std::endl;
			std::cout<<"============"<<std::endl;
			marker.pose.position.x = z;
			marker.pose.position.y = -x;
			marker.pose.position.z = -y;
			break;
		}
	}
	pub_arm.publish(marker);
}

int main(int argc, char** argv){
	ros::init(argc, argv, "detect_arm");
	ros::NodeHandle nh, pnh("~");
	detect_arm foo(nh, pnh);
	while(ros::ok()) ros::spinOnce();
	return 0;
}
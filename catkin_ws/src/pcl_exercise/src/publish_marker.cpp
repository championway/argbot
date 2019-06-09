#include "publish_marker.h"

publish_marker::publish_marker(NodeHandle& nh){    

	t = ros::Time::now().toSec();

	shape = visualization_msgs::Marker::LINE_STRIP;
	marker.header.frame_id = "/camera_link";
	marker.header.stamp = ros::Time::now();
	marker.ns = "lines";
	marker.id = 0;
	marker_pub = nh.advertise<visualization_msgs::Marker>("box_marker", 1);
	marker.type = shape;
    marker.scale.x = 0.03;
    marker.scale.y = 0.03;
    marker.scale.z = 0.03;

    marker.color.r = 0.0f;
    marker.color.g = 0.0f;
    marker.color.b = 1.0f;
    marker.color.a = 1.0;
	sub = nh.subscribe("/prediction", 1, &publish_marker::publish_callback,this);

}
void publish_marker::publish_callback(const sensor_msgs::PointCloud2ConstPtr msg){
	pcl::PointCloud<pcl::PointXYZ>::Ptr orginal_pc (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromROSMsg(*msg, *orginal_pc);

	pcl::PointXYZ minPt, maxPt;
	pcl::getMinMax3D (*orginal_pc, minPt, maxPt);
	cout << "Max x: " << maxPt.x << endl;
	cout << "Max y: " << maxPt.y << endl;
	cout << "Max z: " << maxPt.z << endl;
	cout << "Min x: " << minPt.x << endl;
	cout << "Min y: " << minPt.y << endl;
	cout << "Min z: " << minPt.z << endl;
	cout << "------------------" << endl;
	marker.points.clear();
	p1.x = minPt.x;
	p1.y = minPt.y;
	p1.z = minPt.z;

	p2.x = minPt.x;
	p2.y = maxPt.y;
	p2.z = minPt.z;

	p3.x = maxPt.x;
	p3.y = maxPt.y;
	p3.z = minPt.z;

	p4.x = maxPt.x;
	p4.y = minPt.y;
	p4.z = minPt.z;

	p5.x = minPt.x;
	p5.y = minPt.y;
	p5.z = maxPt.z;

	p6.x = minPt.x;
	p6.y = maxPt.y;
	p6.z = maxPt.z;

	p7.x = maxPt.x;
	p7.y = maxPt.y;
	p7.z = maxPt.z;

	p8.x = maxPt.x;
	p8.y = minPt.y;
	p8.z = maxPt.z;

	marker.points.push_back(p1);
	marker.points.push_back(p2);
	marker.points.push_back(p3);
	marker.points.push_back(p4);
	marker.points.push_back(p1);
	marker.points.push_back(p5);
	marker.points.push_back(p6);
	marker.points.push_back(p7);
	marker.points.push_back(p8);
	marker.points.push_back(p5);
	marker.points.push_back(p6);
	marker.points.push_back(p2);
	marker.points.push_back(p3);
	marker.points.push_back(p7);
	marker.points.push_back(p8);
	marker.points.push_back(p4);

	marker.lifetime = ros::Duration(1.8);
	marker_pub.publish(marker);
}

int main( int argc, char** argv ){
	ros::init(argc, argv, "publish_marker");
	ros::NodeHandle nh;
	publish_marker publish_marker(nh);
	ros::spin(); 
}
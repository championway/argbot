#include <ros/ros.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "realsense2_camera/Extrinsics.h"
// create folder
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "bb_pointnet/pointnet_prediction.h"
#include "bb_pointnet/bb_input.h"
//

using namespace ros;
using namespace std;
using namespace cv;
using namespace pcl;
using namespace message_filters;

static int colormap[5][3] = {{0,0,0}, {255,0,0}, {0,255,0}, {255,0,255}, {0,255,255}};
class mask_to_point{
  public:
    mask_to_point();
    void get_msg();
    void callback(const bb_pointnet::bb_input);
    void getXYZ(float* , float* ,float );
  private:
  	Publisher pc_pub_bbox;
  	ros::Subscriber ssd_result;

    PointCloud<PointXYZRGB>::Ptr pc;
  	float fx;
  	float fy;
  	float cx;
  	float cy;


  	float Projection[3][4];
  	float extrinsics[3][4];
};
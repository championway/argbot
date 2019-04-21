#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/conditional_removal.h>
#include <Eigen/Dense>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <fstream>
#include <vector>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

ros::Publisher pub;   
using namespace ros;
using namespace pcl;
using namespace std;
using namespace cv;

const double PI  =3.141592653589793238463;


class pc_preprocessing{
  public:
    pc_preprocessing();
    void get_msg();
    void callback(const sensor_msgs::ImageConstPtr&,const sensor_msgs::ImageConstPtr&);    // sensor_msgs::PointCloud2
    void getXYZ(float* , float* ,float );
  private:
  	Publisher pc_pre;
  	ros::Subscriber pc_sub;
    PointCloud<PointXYZRGB>::Ptr pc;
    PointCloud<PointXYZRGB>::Ptr pc_filter;
    PointCloud<PointXYZRGB>::Ptr pc_filter_2;
    pcl::ConditionalRemoval<PointXYZRGB> condrem;
    pcl::ConditionAnd<PointXYZRGB>::Ptr range_cond;
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    pcl::SACSegmentation<PointXYZRGB> seg;
    pcl::PointIndices::Ptr inliers;
    pcl::ModelCoefficients::Ptr coefficients;

    message_filters::Subscriber<sensor_msgs::Image> img_sub;
    message_filters::Subscriber<sensor_msgs::Image> depth_sub;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

  	float fx;
  	float fy;
  	float cx;
  	float cy;
    double time;
};
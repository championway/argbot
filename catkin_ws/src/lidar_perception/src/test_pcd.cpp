#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/correspondence.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <stdlib.h>
#include <time.h> 
#include <math.h>
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

int main(int argc, char** argv){
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<PointCloudLabel>::Ptr cloud(new pcl::PointCloud<PointCloudLabel>);
	pcl::PointCloud<PointCloudLabel>::Ptr label_cloud(new pcl::PointCloud<PointCloudLabel>);
	// Read a PCD file from disk.
	if (pcl::io::loadPCDFile<PointCloudLabel>(argv[1], *cloud) != 0)
	{
		return -1;
	}

	std::cout << "Size: " << cloud->points.size () << std::endl;
	std::cout << typeid(cloud->points[2].label).name() << std::endl;
	int pre_label = 0;
	for (size_t i = 0; i < cloud->points.size (); ++i)
    {
      if(int(cloud->points[i].label) == 2){
      	cloud->points[i].r = 0;
	    cloud->points[i].g = 0;
	    cloud->points[i].b = 255;
      }
      else if(int(cloud->points[i].label) == 6){
      	cloud->points[i].r = 255;
	    cloud->points[i].g = 0;
	    cloud->points[i].b = 0;
      }
      else if(int(cloud->points[i].label) == 8){
      	cloud->points[i].r = 0;
	    cloud->points[i].g = 255;
	    cloud->points[i].b = 0;
      }
      else{
      	cloud->points[i].r = 255;
      	cloud->points[i].g = 255;
	    cloud->points[i].b = 255;
      }
    }
    srand( time(NULL) );
    int a[3][2];
    std::cout << sqrt(pow(4.0, 2)) << std::endl;
    std::string test[3] = {"aa","bbb","ccc"};
    label_cloud->points.resize (2);
  	label_cloud->width = 2;
  	label_cloud->height = 1;
  	label_cloud->points[0].x = label_cloud->points[0].y = label_cloud->points[0].z = 1;
	label_cloud->points[1].x = label_cloud->points[2].y = label_cloud->points[3].z = 5;
	//std::cout << cloud->points[2].z << std::endl;
	//std::cout << int(cloud->points[2].rgba) << std::endl;
	//std::cout << int(cloud->points[2].g) << std::endl;
	//std::cout << int(cloud->points[2].b) << std::endl;
	//std::cout << cloud->points[2].label << std::endl;
	//std::cout << typeid(cloud->points[2].r).name() << std::endl;
	//std::cout << cloud->points[2].g << std::endl;
	//std::cout << cloud->points[2].b << std::endl;
  	/*float sum_x=0, sum_y=0, sum_z=0;
	for (size_t i = 0; i < cloud->points.size (); ++i)
  	{
      sum_x += cloud->points[i].x;
      sum_y += cloud->points[i].y;
      sum_z += cloud->points[i].z;
  	}
    sum_x /= cloud->points.size ();
    sum_y /= cloud->points.size ();
    sum_z /= cloud->points.size ();*/
	//std::cout << "origin size: " << cloud->points.size () << std::endl;
	//std::vector<int> indices;
	//pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
	//std::cout << "size: " << cloud->points.size () << std::endl;
  	/*for (size_t i = 0; i < cloud->points.size (); ++i)
    {
      cloud->points[i].x -= sum_x;
      cloud->points[i].y -= sum_y;
      cloud->points[i].z -= sum_z;
      
      cloud->points[i].r = 255;
      cloud->points[i].g = 0;
      cloud->points[i].b = 0;
    }*/
    std::cout << "finish" << std::endl;
  
  	pcl::io::savePCDFileASCII ("Hi.pcd", *label_cloud);
  	std::cerr << "Saved " << cloud->points.size () << " data points to Hi.pcd." << std::endl;

}

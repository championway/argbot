#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, PoseStamped, Point
import rospkg
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import message_filters

class D435():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.bridge = CvBridge()
		#self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.img_cb, queue_size = 1, buff_size = 2**24)
		
		self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
		self.image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
		self.ts = message_filters.ApproximateTimeSynchronizer([self.depth_sub, self.image_sub], 1, 0.3)
		self.ts.registerCallback(self.img_cb)

		self.count = 0

	def img_cb(self, depth_data, rgb_data):
		cv_depthimage = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
		cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
		print(cv_depthimage.dtype, cv_image.dtype)
		cv_depthimage = np.array(cv_depthimage, dtype=np.uint16)
		cv2.imwrite('/media/arg_ws3/5E703E3A703E18EB/data/d435_mm/empty/' + str(self.count) + '_depth.png', cv_depthimage)
		cv2.imwrite('/media/arg_ws3/5E703E3A703E18EB/data/d435_mm/empty/' + str(self.count) + '_rgb.png', cv_image)
		rospy.loginfo('save image')
		self.count = self.count + 1
		rospy.sleep(1)

if __name__ == '__main__':
	rospy.init_node('D435')
	foo = D435()
	rospy.spin()
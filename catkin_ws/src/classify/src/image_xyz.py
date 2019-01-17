#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker, MarkerArray
from robotx_msgs.msg import PCL_points, ObstaclePose, ObstaclePoseList
import rospkg
from cv_bridge import CvBridge, CvBridgeError

class pcl2img():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/pcl_points', PCL_points, self.call_back)
		self.pub_marker = rospy.Publisher("/obs_classify", MarkerArray, queue_size = 1)
		#rospy.Subscriber('/pcl_array', PoseArray, self.call_back)
		self.boundary = 50
		self.height = self.width = 480.0
		self.point_size = 4	# must be integer
		self.image = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		self.index = 0

	def call_back(self, msg):
		cluster_num = len(msg.list)
		#pcl_size = len(msg.poses)
		for i in range(cluster_num):
			self.image = np.zeros((int(self.height), int(self.width), 3), np.uint8)
			img = []
			pcl_size = len(msg.list[i].poses)
			avg_x = avg_y = avg_z = 0
			for j in range(pcl_size): # project to XY, YZ, XZ plane
				avg_x = avg_x + msg.list[i].poses[j].position.x
				avg_y = avg_y + msg.list[i].poses[j].position.y
				avg_z = avg_z + msg.list[i].poses[j].position.z
				img.append([msg.list[i].poses[j].position.x, msg.list[i].poses[j].position.y, msg.list[i].poses[j].position.z])
			self.toIMG(pcl_size, img, 'img')
			cv2.imwrite( "Image_" + str(self.index) + ".jpg", self.image)
			self.index = self.index + 1
			print "Save image"

	def toIMG(self, pcl_size, pcl_array, plane):
		min_x = 10e5
		min_y = 10e5
		min_z = 10e5
		max_x = -10e5
		max_y = -10e5
		max_z = -10e5
		for i in range(pcl_size):
			if min_x > pcl_array[i][0]:
				min_x = pcl_array[i][0]
			if min_y > pcl_array[i][1]:
				min_y = pcl_array[i][1]
			if min_z > pcl_array[i][2]:
				min_z = pcl_array[i][2]
			if max_x < pcl_array[i][0]:
				max_x = pcl_array[i][0]
			if max_y < pcl_array[i][1]:
				max_y = pcl_array[i][1]
			if max_z < pcl_array[i][2]:
				max_z = pcl_array[i][2]
		x_size = max_x - min_x
		y_size = max_y - min_y
		z_size = max_z - min_z
		z_norm = None
		if z_size != 0:
			z_norm = 254.0/z_size
		max_size = None
		min_size = None
		shift_m = False
		shift_n = False
		if x_size > y_size:
			max_size = x_size
			min_size = y_size
			shift_n = True
		else:
			max_size = y_size
			min_size = x_size
			shift_m = True
		scale = float((self.height-self.boundary*2)/max_size)
		shift_size = int(round((self.height - self.boundary*2 - min_size*scale)/2))
		img = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		for i in range(pcl_size):
			if shift_m:
				pcl_array[i][0] = int(round((pcl_array[i][0] - min_x)*scale)) + shift_size + self.boundary
				pcl_array[i][1] = int(round((pcl_array[i][1] - min_y)*scale)) + self.boundary
			elif shift_n:
				pcl_array[i][0] = int(round((pcl_array[i][0] - min_x)*scale)) + self.boundary
				pcl_array[i][1] = int(round((pcl_array[i][1] - min_y)*scale)) + shift_size + self.boundary
			for m in range(-self.point_size, self.point_size + 1):
				for n in range(-self.point_size, self.point_size + 1):
					if z_norm != None:
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][0] = int(round(pcl_array[i][2]*z_norm))
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][1] = int(round(pcl_array[i][2]*z_norm))
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][2] = int(round(pcl_array[i][2]*z_norm))
					else:
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][0] = 255
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][2] = 255
						self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][1] = 255
		#cv2.imwrite( "Image_" + plane + ".jpg", img )

if __name__ == '__main__':
	rospy.init_node('pcl2img')
	foo = pcl2img()
	rospy.spin()
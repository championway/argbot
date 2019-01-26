#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
from sensor_msgs.msg import Image, LaserScan
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData
import rospkg
from cv_bridge import CvBridge, CvBridgeError

class mapping():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber("/pcl_points", PoseArray, self.call_back, queue_size=1)
		self.pub_map = rospy.Publisher('/local_map', OccupancyGrid, queue_size = 1)
		self.resolution = 0.5
		self.width = 100
		self.height = 100
		self.origin = Pose()
		self.local_map = OccupancyGrid()
		self.dilating_size = 2
		self.click_pt = None
		self.if_click = False

	def init_param(self):
		self.occupancygrid = np.zeros((self.height, self.width))
		self.local_map = OccupancyGrid()
		self.local_map.info.resolution = self.resolution
		self.local_map.info.width = self.width
		self.local_map.info.height = self.height
		self.origin.position.x = -self.width*self.resolution/2.
		self.origin.position.y = -self.height*self.resolution/2.
		self.local_map.info.origin = self.origin

	def cb_rviz(self, msg):
		self.click_pt = [msg.pose.position.x, msg.pose.position.y]
		self.publish_topic()

	def call_back(self, msg):
		self.init_param()
		self.local_map.header = msg.header
		for i in range(len(msg.poses)):
			p = (msg.poses[i].position.x, msg.poses[i].position.y)
			x, y = self.map2occupancygrid(p)
			width_in_range = (x >= self.width - self.dilating_size or x <= self.dilating_size)
			height_in_range = (y >= self.height - self.dilating_size or y <= self.dilating_size)
			if width_in_range or height_in_range:
				continue # To prevent point cloud range over occupancy grid range
			self.occupancygrid[y][x] = 100

		# map dilating
		for i in range(self.height):
			for j in range(self.width):
				if self.occupancygrid[i][j] == 100:
					for m in range(-self.dilating_size, self.dilating_size + 1):
						for n in range(-self.dilating_size, self.dilating_size + 1):
							if self.occupancygrid[i+m][j+n] != 100:
								self.occupancygrid[i+m][j+n] = 50

		for i in range(self.height):
			for j in range(self.width):
				self.local_map.data.append(self.occupancygrid[i][j])
		self.pub_map.publish(self.local_map)

	def occupancygrid2map(self, p):
		x = p[0]*self.resolution + self.origin.position.x
		y = p[1]*self.resolution + self.origin.position.y
		return [x, y]

	def map2occupancygrid(self, p):
		x = int((p[0]-self.origin.position.x)/self.resolution)
		y = int((p[1]-self.origin.position.y)/self.resolution)
		return [x, y]

if __name__ == '__main__':
	rospy.init_node('mapping')
	foo = mapping()
	rospy.spin()
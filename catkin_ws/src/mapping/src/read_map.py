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

class read_map():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/map', OccupancyGrid, self.call_back, queue_size = 1, buff_size = 2**24)
		rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.cb_rviz, queue_size = 1)
		self.pub_map = rospy.Publisher('/new_map', OccupancyGrid, queue_size = 1)
		self.resolution = None
		self.width = None
		self.height = None
		self.origin = None
		self.new_map = OccupancyGrid()
		self.click_pt = None
		self.if_click = False

	def init_param(self):
		self.occupancygrid = np.zeros((self.height, self.width))

	def cb_rviz(self, msg):
		self.click_pt = [msg.pose.position.x, msg.pose.position.y]
		self.publish_topic()

	def call_back(self, msg):
		self.resolution = msg.info.resolution
		self.width = int(msg.info.width)
		self.height = int(msg.info.height)
		self.origin = msg.info.origin
		self.init_param() 
		for i in range(self.height):
			for j in range(self.width):
				self.occupancygrid[i][j] = msg.data[i*self.width + j]
		h, w = self.occupancygrid.shape
		self.new_map.header = msg.header
		self.new_map.info = msg.info
		for i in range(h):
			for j in range(w):
				self.new_map.data.append(self.occupancygrid[i][j])

	def publish_topic(self):
		x = int((self.click_pt[0]-self.origin.position.x)/self.resolution)
		y = int((self.click_pt[1]-self.origin.position.y)/self.resolution)
		print(x, y)
		print(self.click_pt)
		value = self.occupancygrid[y][x]
		print(value)

if __name__ == '__main__':
	rospy.init_node('read_map')
	foo = read_map()
	rospy.spin()
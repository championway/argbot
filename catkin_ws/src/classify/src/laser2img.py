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
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import rospkg
from cv_bridge import CvBridge, CvBridgeError

class laser2img():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/X1/scan', LaserScan, self.call_back, queue_size = 1, buff_size = 2**24)

		self.bin = 360
		self.range_max = 5.5
		self.border = 5
		self.scale = 10.
		self.point_size = 2

	def init_param(self):
		self.width = int(self.range_max*self.scale*2 + self.border*2)
		self.height = int(self.range_max*self.scale*2 + self.border*2)
		self.img = np.zeros((int(self.height), int(self.width)), np.uint8)

	def call_back(self, msg):
		self.range_max = msg.range_max
		self.bin = len(msg.ranges)
		self.init_param()

		for i in range(self.bin):
			if msg.ranges[i] != float("inf"):
				rad = (i/360.)*2.*math.pi/2.
				x = self.scale*msg.ranges[i] * np.cos(rad)
				y = self.scale*msg.ranges[i] * np.sin(rad)
				x_ = int(x + self.width/2.)
				y_ = int(y + self.height/2.)
				self.img[x_][y_] = 255
				self.img = self.img_dilate(self.img, x_, y_)
		cv2.imwrite( "Image.jpg", self.img)

	def img_dilate(self, img, x, y):
		for m in range(-self.point_size, self.point_size + 1):
			for n in range(-self.point_size, self.point_size + 1):
				img[x + m][y + n] = 255
		return img

if __name__ == '__main__':
	rospy.init_node('laser2img')
	foo = laser2img()
	rospy.spin()
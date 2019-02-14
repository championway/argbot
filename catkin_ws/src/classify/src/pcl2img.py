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
		rospy.Subscriber('/scan', LaserScan, self.call_back, queue_size = 1, buff_size = 2**24)

		self.bin = 360
		self.range_max = 5.5
		self.border = 10
		self.scale = 50.

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
				rad = (i/360.)*2.*math.pi
				x = self.scale*msg.ranges[i] * np.cos(rad)
				y = self.scale*msg.ranges[i] * np.sin(rad)
				self.img[int(x + self.width/2.)][int(y + self.height/2.)] = 255
		cv2.imwrite( "Image.jpg", self.img )
		#print('Save image')

if __name__ == '__main__':
	rospy.init_node('laser2img')
	foo = laser2img()
	rospy.spin()
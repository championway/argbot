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

class position_record():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/scan', LaserScan, self.call_back, queue_size = 1, buff_size = 2**24)
		self.OLD_POSE_NUM = 20
		self.old_pos = []	# record old positions with numbers of OLD_POSE_NUM
		self.cur_pos = None # robot current position

	def init_param(self):
		pass

	def call_back(self, msg):
		self.init_param()

		# get current position
		self.cur_pos[0] = msg.position.x
		self.cur_pos[1] = msg.position.y

		self.shift_coordinate()
		self.update_pos() # update self.old_pos

	def shift_coordinate(self):
		# the previous position
		pre_pos = self.old_pos[-1]

		# get the shift of the coordinate
		shift_x = self.cur_pos[0] - pre_pos[0]
		shify_y = self.cur_pos[1] - pre_pos[1]

		# shift the coordinate of old positions
		for idx in range(len(self.old_pos)):
			self.old_pos[idx][0] = self.old_pos[idx][0] - shift_x
			self.old_pos[idx][1] = self.old_pos[idx][1] - shift_y

	def update_pos(self):
		self.old_pos.append([0, 0])
		self.old_pos = self.old_pos[-self.OLD_POSE_NUM:] # We only want to record certain amount of old position

if __name__ == '__main__':
	rospy.init_node('position_record')
	foo = position_record()
	rospy.spin()
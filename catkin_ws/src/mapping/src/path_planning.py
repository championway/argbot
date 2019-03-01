#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
import random
from sensor_msgs.msg import Image, LaserScan
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry, Path
import rospkg
from cv_bridge import CvBridge, CvBridgeError

class navigate():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/map', OccupancyGrid, self.cb_map, queue_size = 1, buff_size = 2**24)
		rospy.Subscriber('/odom', Odometry, self.cb_odom, queue_size = 1)
		rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.cb_path, queue_size = 1)
		self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size = 1)
		self.resolution = None
		self.width = None
		self.height = None
		self.origin = None
		self.click_pt = None
		self.if_click = False
		self.arr_threshold = 0.5
		self.goal_distance = 3
		self.old_goal = None
		self.goal = None
		self.get_map = False
		self.pub_counter = 0

	def init_param(self):
		self.occupancygrid = np.zeros((self.height, self.width))

	def update_goal(self):
		rospy.loginfo("Update goal")
		good = False
		while(not good):
			self.goal = self.occupancygrid2map(self.get_goal())
			if self.old_goal is None:
				good = True
			elif self.distance(self.old_goal, self.goal) > self.goal_distance:
				good = True
		self.old_goal = self.goal[:]

	def cb_path(self, msg):
		if len(msg.poses) == 0:
			print("Wrong goal")
			self.update_goal()

	def cb_map(self, msg):
		self.resolution = msg.info.resolution
		self.width = int(msg.info.width)
		self.height = int(msg.info.height)
		self.origin = msg.info.origin
		self.init_param() 
		for i in range(self.height):
			for j in range(self.width):
				self.occupancygrid[i][j] = msg.data[i*self.width + j]
		self.get_map = True
		
	def cb_odom(self, msg):
		pose = [msg.pose.pose.position.x, msg.pose.pose.position.y]
		if not self.get_map:
			return
		if self.goal is None:
			self.update_goal()
		if self.distance(pose, self.goal) < self.arr_threshold :
			self.update_goal()

		if self.pub_counter < 10:
			goal_pose = PoseStamped()
			goal_pose.header.frame_id = "map"
			goal_pose.pose.position.x = self.goal[0]
			goal_pose.pose.position.y = self.goal[1]
			goal_pose.pose.orientation.w = 1
			self.pub_goal.publish(goal_pose)
			self.pub_counter = self.pub_counter + 1
		else:
			self.pub_counter = 11

	def get_goal(self):
		self.pub_counter = 0
		occupied = True
		x = None
		y = None
		while(True):
			x = random.randint(0, self.width -1)
			y = random.randint(0, self.height -1)
			if self.occupancygrid[y][x] == 0:
				break
		return [x, y]

	def distance(self, p1, p2):
		return math.sqrt((p1[0]-p2[0])**2 + (p1[1] - p2[1])**2)

	def occupancygrid2map(self, p):
		x = p[0]*self.resolution + self.origin.position.x
		y = p[1]*self.resolution + self.origin.position.y
		return [x, y]

	def map2occupancygrid(self, p):
		x = int((p[0]-self.origin.position.x)/self.resolution)
		y = int((p[1]-self.origin.position.y)/self.resolution)
		return [x, y]	

if __name__ == '__main__':
	rospy.init_node('navigate')
	foo = navigate()
	rospy.spin()
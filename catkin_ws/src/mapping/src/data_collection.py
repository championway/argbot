#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
import os
import random
from sensor_msgs.msg import Image, LaserScan
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry, Path
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber

class navigate():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/map', OccupancyGrid, self.cb_map, queue_size = 1, buff_size = 2**24)
		odom_sub = Subscriber('/odom', Odometry, queue_size = 1)
		scan_sub = Subscriber('/scan', LaserScan, queue_size = 1)
		tss = ApproximateTimeSynchronizer([odom_sub, scan_sub], 1, 0.1)
		tss.registerCallback(self.call_back)
		rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.cb_path, queue_size = 1)
		self.pub_goal = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size = 1)
		self.resolution = None
		self.width = None
		self.height = None
		self.origin = None
		self.click_pt = None
		self.if_click = False
		self.arr_threshold = 0.5
		self.goal_distance = 2.4
		self.old_goal = None
		self.goal = None
		self.get_map = False
		self.pub_counter = 0

		self.bin = 360
		self.range_max = 5.5
		self.border = 3
		self.scale = 180.
		self.point_size = 2

		# For data writing
		self.SAVE_ROOT = "/media/arg_ws3/TOSHIBA EXT/data/trajectory/"
		self.root_file = open(self.SAVE_ROOT + "root.txt", "w")
		self.start_list = []
		self.start_frame = ""
		self.frame_len = 0
		self.episode_counter = 0
		self.time_old = rospy.get_time()
		self.time_now = rospy.get_time()
		self.DATA_PREFIX = "stage4_"
		self.SAVE_PATH_IMG = self.SAVE_ROOT + "images/"
		self.SAVE_PATH_ANN = self.SAVE_ROOT + "annotations/"
		self.FILE_NAME = ""
		self.COUNTER = 0
		self.NEXT_FRAME = ""
		self.IS_FIRST_FRAME = "True"
		self.ORIGIN_X = ""
		self.ORIGIN_Y = ""
		self.ROBOT_X = ""
		self.ROBOT_Y = ""
		self.TIME_STAMP = ""
		self.IMG_NAME = ""
		self.SCALE = str(self.scale)

		if not os.path.exists(self.SAVE_PATH_IMG):
			os.makedirs(self.SAVE_PATH_IMG)
		if not os.path.exists(self.SAVE_PATH_ANN):
			os.makedirs(self.SAVE_PATH_ANN)

	def writing_data(self):
		if self.IS_FIRST_FRAME == "True":
			self.start_frame = self.FILE_NAME
		self.IMG_NAME = self.FILE_NAME
		file = open(self.SAVE_PATH_ANN + self.FILE_NAME + ".txt", "w")
		data = ""
		data = data + self.IS_FIRST_FRAME + "\n"
		data = data + self.NEXT_FRAME + "\n"
		data = data + self.ORIGIN_X + "\n"
		data = data + self.ORIGIN_Y + "\n"
		data = data + self.ROBOT_X + "\n"
		data = data + self.ROBOT_Y + "\n"
		data = data + self.TIME_STAMP + "\n"
		data = data + self.IMG_NAME + "\n"
		data = data + self.SCALE
		file.write(data)
		file.close()
		cv2.imwrite(self.SAVE_PATH_IMG + self.IMG_NAME + ".jpg", self.img)
		self.IS_FIRST_FRAME = "False"
		self.frame_len = self.frame_len + 1

	def initial_scan(self):
		self.scan_width = int(self.range_max*self.scale*2 + self.border*2)
		self.scan_height = int(self.range_max*self.scale*2 + self.border*2)
		self.img_width = 300
		self.img_height = 300
		self.img = np.zeros((int(self.img_height), int(self.img_width)), np.uint8)

	def init_param(self):
		self.occupancygrid = np.zeros((self.height, self.width))

	def update_goal(self):
		if self.ORIGIN_X != "":
			self.start_list.append([self.start_frame, self.frame_len])
			self.root_file.write(str(self.start_frame + "," + str(self.frame_len) + "\n"))
		good = False
		while(not good):
			self.goal = self.occupancygrid2map(self.get_goal())
			if self.old_goal is None:
				good = True
			elif self.distance(self.old_goal, self.goal) > self.goal_distance:
				good = True
		self.old_goal = self.goal[:]
		self.IS_FIRST_FRAME = "True"
		self.frame_len = 0
		self.ORIGIN_X = str(self.goal[0])
		self.ORIGIN_Y = str(self.goal[1])
		self.episode_counter = self.episode_counter + 1
		output_str = str(self.episode_counter), ": Update goal"
		rospy.loginfo(output_str)

	def cb_path(self, msg):
		if len(msg.poses) == 0:
			print("Wrong goal")
			self.episode_counter = self.episode_counter - 1
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

	def img_dilate(self, img, x, y):
		for m in range(-self.point_size, self.point_size + 1):
			for n in range(-self.point_size, self.point_size + 1):
				if x+m < self.img_width and y+n < self.img_height:
					img[y + n][x + m] = 255
		return img
		
	def call_back(self, msg_odom, msg_scan):
		self.time_now = rospy.get_time()
		self.range_max = msg_scan.range_max
		self.bin = len(msg_scan.ranges)
		self.initial_scan()

		for i in range(self.bin):
			if msg_scan.ranges[i] != float("inf"):
				rad = (i/360.)*2.*math.pi
				x = self.scale*msg_scan.ranges[i] * np.sin(rad)
				y = self.scale*msg_scan.ranges[i] * np.cos(rad)
				x_ = int(x + self.img_width/2.)
				y_ = int(y + self.img_height/2.)
				if x_ < self.img_height and y_ < self.img_width:
					if x_ >= 0 and y_ >= 0:
						self.img[y_][x_] = 255
						self.img = self.img_dilate(self.img, x_, y_)

		pose = [msg_odom.pose.pose.position.x, msg_odom.pose.pose.position.y]
		if not self.get_map:
			return
		if self.time_now - self.time_old >= 0.5 and self.ORIGIN_X != "":
			self.TIME_STAMP = str(self.time_now)
			self.ROBOT_X = str(pose[0])
			self.ROBOT_Y = str(pose[1])
			self.FILE_NAME = self.DATA_PREFIX + str(self.COUNTER)
			self.COUNTER = self.COUNTER + 1
			self.NEXT_FRAME = self.DATA_PREFIX + str(self.COUNTER)
			self.writing_data()
			self.time_old = rospy.get_time()

		if self.goal is None:
			self.update_goal()
		if self.distance(pose, self.goal) < self.arr_threshold :
			self.update_goal()

		#if self.episode_counter > 100:
		#	rospy.on_shutdown(self.shutdown)

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

	def shutdown(self):
  		rospy.loginfo("Finish data collection!!!")

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
	node = navigate()
	rospy.spin()
	if rospy.is_shutdown():
		print("")
		rospy.loginfo("Shutdown!!!!!!!!!!!!")
		rospy.loginfo("Writing File......")
		node.root_file.close()
#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
from tf import TransformListener,TransformerROS
from tf import LookupException, ConnectivityException, ExtrapolationException
import message_filters
from message_filters import ApproximateTimeSynchronizer, TimeSynchronizer
import struct
import math
import time
from sensor_msgs.msg import Image, LaserScan
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from path_planning import AStar

class mapping():
	def __init__(self):
		self.node_name = rospy.get_name()
		self.tf = TransformListener()
		self.transformer = TransformerROS()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		#rospy.Subscriber("/pcl_points", PoseArray, self.call_back, queue_size=1, buff_size = 2**24)
		sub_pcl = message_filters.Subscriber("/pcl_points", PoseArray)
		sub_odom = message_filters.Subscriber("/odometry/ground_truth", Odometry)
		ats = ApproximateTimeSynchronizer((sub_pcl, sub_odom), queue_size = 1, slop = 0.1)
		ats.registerCallback(self.call_back)
		rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb_new_goal, queue_size=1)
		self.pub_map = rospy.Publisher('/local_map', OccupancyGrid, queue_size = 1)
		self.pub_rviz = rospy.Publisher("/wp_line", Marker, queue_size = 1)
		self.pub_poses = rospy.Publisher("/path_points", PoseArray, queue_size = 1)
		self.resolution = 0.25
		self.robot = Pose()
		self.width = 200
		self.height = 200
		self.origin = Pose()
		self.local_map = OccupancyGrid()
		self.dilating_size = 6
		self.wall_width = 3
		self.start_planning = False
		self.transpose_matrix = None
		self.goal = []
		self.astar = AStar()
		self.msg_count = 0
		self.planning_range = 20
		self.frame_id = "map"

	def init_param(self):
		self.occupancygrid = np.zeros((self.height, self.width))
		self.local_map = OccupancyGrid()
		self.local_map.info.resolution = self.resolution
		self.local_map.info.width = self.width
		self.local_map.info.height = self.height
		self.origin.position.x = -self.width*self.resolution/2. + self.robot.position.x
		self.origin.position.y = -self.height*self.resolution/2. + self.robot.position.y
		self.local_map.info.origin = self.origin

	def cb_rviz(self, msg):
		self.click_pt = [msg.pose.position.x, msg.pose.position.y]
		self.publish_topic()

	def call_back(self, pcl_msg, odom_msg):
		self.robot = odom_msg.pose.pose
		self.msg_count = self.msg_count + 1
		self.init_param()
		self.local_map.header = pcl_msg.header
		self.local_map.header.frame_id = self.frame_id
		self.get_tf()
		if self.transpose_matrix is None:
			return
		for i in range(len(pcl_msg.poses)):
			p = (pcl_msg.poses[i].position.x, pcl_msg.poses[i].position.y, pcl_msg.poses[i].position.z, 1)
			local_p = np.array(p)
			global_p = np.dot(self.transpose_matrix, local_p)
			x, y = self.map2occupancygrid(global_p)
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
								if m > self.wall_width or m < -self.wall_width or n > self.wall_width or n < -self.wall_width:
									if self.occupancygrid[i+m][j+n] != 90:
										self.occupancygrid[i+m][j+n] = 50
								else:
									self.occupancygrid[i+m][j+n] = 90

		for i in range(self.height):
			for j in range(self.width):
				self.local_map.data.append(self.occupancygrid[i][j])
		self.pub_map.publish(self.local_map)

		if self.start_planning:
			self.path_planning()

	def get_tf(self):
		try:
			position, quaternion = self.tf.lookupTransform( "/map", "/X1/front_laser",rospy.Time(0))
			self.transpose_matrix = self.transformer.fromTranslationRotation(position, quaternion)
		except (LookupException, ConnectivityException, ExtrapolationException):
			print("Nothing Happen")

	def path_planning(self):
		if self.msg_count % 5 != 0:
			return
		self.msg_count = 0
		cost_map = np.zeros((self.height, self.width))
		border = self.planning_range/self.resolution
		h_min = int((self.height - border)/2.)
		h_max = int(self.height - (self.height - border)/2.)
		w_min = int((self.height - border)/2.)
		w_max = int(self.width - (self.width - border)/2.)
		for i in range(self.width):
			for j in range(self.width):
				if i > h_min and i < h_max:
					if j > w_min and j < w_max:
						cost_map[i][j] = self.occupancygrid[i][j]
		start_point = self.map2occupancygrid((self.robot.position.x, self.robot.position.y))
		start = (start_point[1], start_point[0])
		goal = self.map2occupancygrid(self.goal)
		end = (goal[1], goal[0])
		self.astar.initial(cost_map, start, end)
		path = self.astar.planning()
		self.pub_path(path)
		self.rviz(path)

	def cb_new_goal(self, p):
		self.goal = [p.pose.position.x, p.pose.position.y]
		self.start_planning = True

	def occupancygrid2map(self, p):
		x = p[0]*self.resolution + self.origin.position.x + self.resolution/2.
		y = p[1]*self.resolution + self.origin.position.y + self.resolution/2.
		return [x, y]

	def map2occupancygrid(self, p):
		x = int((p[0]-self.origin.position.x)/self.resolution)
		y = int((p[1]-self.origin.position.y)/self.resolution)
		return [x, y]

	def pub_path(self, path):
		poses = PoseArray()
		for i in range(len(path)):
			p = self.occupancygrid2map([path[i][1], path[i][0]])
			pose = Pose()
			pose.position.x = p[0]
			pose.position.y = p[1]
			pose.position.z = 0
			poses.poses.append(pose)
		self.pub_poses.publish(poses)
			
	def rviz(self, path):
		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.type = marker.LINE_STRIP
		marker.action = marker.ADD
		marker.scale.x = 0.3
		marker.scale.y = 0.3
		marker.scale.z = 0.3
		marker.color.a = 1.0
		marker.color.r = 1.0
		marker.color.g = 0.
		marker.color.b = 0.
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = 0.0
		marker.pose.position.y = 0.0
		marker.pose.position.z = 0.0
		marker.points = []
		for i in range(len(path)):
			p = self.occupancygrid2map([path[i][1], path[i][0]])
			point = Point()
			point.x = p[0]
			point.y = p[1]
			point.z = 0
			marker.points.append(point)
		self.pub_rviz.publish(marker)

if __name__ == '__main__':
	rospy.init_node('mapping')
	foo = mapping()
	rospy.spin()
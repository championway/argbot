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
from tf import TransformListener,TransformerROS
from tf import LookupException, ConnectivityException, ExtrapolationException
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from path_planning import AStar

class mapping():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber("/pcl_points", PoseArray, self.call_back, queue_size=1, buff_size = 2**24)
		rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.cb_new_goal, queue_size=1)
		self.pub_map = rospy.Publisher('/local_map', OccupancyGrid, queue_size = 1)
		self.pub_planning_map = rospy.Publisher('/planning_map', OccupancyGrid, queue_size = 1)
		self.pub_rviz = rospy.Publisher("/wp_line", Marker, queue_size = 1)
		self.pub_path = rospy.Publisher("/planning_path", Path, queue_size = 1)
		self.resolution = 0.25
		self.width = 200
		self.height = 200
		self.origin = Pose()
		self.local_map = OccupancyGrid()
		self.dilating_size = 6
		self.wall_width = 3
		self.start_planning = False
		self.goal = []
		self.goal_occupancygrid = []
		self.astar = AStar()
		self.msg_count = 0
		self.border = 50
		self.frame_id = None
		self.map_frame = "/map"
		self.robot_pose = None
		self.static_local = False
		self.transpose_matrix = None
		self.old_transform_matrix = None
		self.pre_path = Path()
		self.radius = 5

	def init_param(self):
		self.occupancygrid = np.zeros((self.height, self.width))
		self.local_map = OccupancyGrid()
		self.local_map.info.resolution = self.resolution
		self.local_map.info.width = self.width
		self.local_map.info.height = self.height
		self.origin.position.x = -self.width*self.resolution/2.
		self.origin.position.y = -self.height*self.resolution/2.
		self.local_map.info.origin = self.origin

		self.cost_map = np.zeros((self.height, self.width))
		self.planning_map = OccupancyGrid()
		self.planning_map.info.resolution = self.resolution
		self.planning_map.info.width = self.width
		self.planning_map.info.height = self.height
		self.planning_map.info.origin = self.origin

	def cb_rviz(self, msg):
		self.click_pt = [msg.pose.position.x, msg.pose.position.y]
		self.publish_topic()

	def call_back(self, msg):
		self.msg_count = self.msg_count + 1
		self.init_param()
		self.frame_id = msg.header.frame_id
		self.local_map.header = msg.header
		self.planning_map.header = msg.header
		try:
			position, quaternion = tf_.lookupTransform(self.map_frame, msg.header.frame_id, rospy.Time(0))
			self.transpose_matrix = transformer.fromTranslationRotation(position, quaternion)
			self.robot_pose = np.dot(self.transpose_matrix, [0, 0, 0, 1])
			self.origin.position.x =  -self.width*self.resolution/2. + self.robot_pose[0]
			self.origin.position.y =  -self.height*self.resolution/2. + self.robot_pose[1]
			self.local_map.info.origin = self.origin
			if True or self.goal!= []:
				#print(self.dx, self.dy)# 15, 7
				#goal_temp = np.dot(self.transpose_matrix, [self.dx, self.dy, 0, 1])
				#self.goal_occupancygrid = self.map2occupancygrid(self.goal)
				self.start_planning = True
			for i in range(len(msg.poses)):
				origin_p = np.array([msg.poses[i].position.x, msg.poses[i].position.y, msg.poses[i].position.z, 1])
				new_p = np.dot(self.transpose_matrix, origin_p)
				p = (new_p[0], new_p[1])
				x, y = self.map2occupancygrid(p)
				width_in_range = (x >= self.width - self.dilating_size or x <= self.dilating_size)
				height_in_range = (y >= self.height - self.dilating_size or y <= self.dilating_size)
				if width_in_range or height_in_range:
					continue # To prevent point cloud range over occupancy grid range
				self.occupancygrid[y][x] = 100
		except (LookupException, ConnectivityException, ExtrapolationException):
			return

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
		self.local_map.header.frame_id = self.map_frame
		self.planning_map.header.frame_id = self.map_frame
		self.pub_map.publish(self.local_map)

		if self.start_planning:
			self.path_planning()

	def get_goal(self):
		sample = np.linspace(-90, 90, 30)
		goal_list = []
		for degree in sample:
			rad = np.deg2rad(degree)
			x = self.radius * np.cos(rad)
			y = self.radius * np.sin(rad)
			p = np.dot(self.transpose_matrix, [x, y, 0, 1])
			goal = self.map2occupancygrid([p[0], p[1]])
			if goal not in goal_list:
				goal_list.append(goal)
		return goal_list


	def path_planning(self):
		if self.msg_count % 5 != 0:
			return
		self.msg_count = 0
		#self.cost_map = np.zeros((self.height, self.width))
		for i in range(self.height):
			for j in range(self.width):
				if i > int(self.border) and i < int(self.height - self.border):
					if j > int(self.border) and j < int(self.width - self.border):
						self.cost_map[i][j] = self.occupancygrid[i][j]
		start_point = self.map2occupancygrid((self.robot_pose[0], self.robot_pose[1]))
		start = (start_point[1], start_point[0])
		goal_list = self.get_goal()
		success = False
		for goal in goal_list:
			end = (goal[1], goal[0])
			self.astar.initial(self.cost_map, start, end)
			path, success = self.astar.planning()
			if success:
				break
		if success:
			self.old_transform_matrix = self.transpose_matrix
			self.pre_path = self.pub_topic(path)
		else:
			print("No")
			old_path = []
			for p in self.pre_path.poses:
				p_occupancygrid = self.map2occupancygrid([p.pose.position.x, p.pose.position.y])
				old_path.append([p_occupancygrid[1], p_occupancygrid[0]])
			_ = self.pub_topic(old_path)

		for i in range(self.height):
			for j in range(self.width):
				self.planning_map.data.append(self.cost_map[i][j])
		self.pub_planning_map.publish(self.planning_map)

	def cb_new_goal(self, p):
		self.goal = [p.pose.position.x, p.pose.position.y]
		# ========== For wall following ==========
		self.dx = self.goal[0] - self.robot_pose[0]
		self.dy = self.goal[1] - self.robot_pose[1]

	def occupancygrid2map(self, p):
		x = p[0]*self.resolution + self.origin.position.x + self.resolution/2.
		y = p[1]*self.resolution + self.origin.position.y + self.resolution/2.
		return [x, y]

	def map2occupancygrid(self, p):
		x = int((p[0]-self.origin.position.x)/self.resolution)
		y = int((p[1]-self.origin.position.y)/self.resolution)
		return [x, y]
			
	def pub_topic(self, path):
		path_msg = Path()
		path_msg.header.frame_id = self.map_frame
		marker = Marker()
		marker.header.frame_id = self.map_frame
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
			pose = PoseStamped()
			pose.pose.position.x = p[0]
			pose.pose.position.y = p[1]
			path_msg.poses.append(pose)
			point = Point()
			point.x = p[0]
			point.y = p[1]
			point.z = 0
			marker.points.append(point)
		self.pub_rviz.publish(marker)
		self.pub_path.publish(path_msg)
		return path_msg

if __name__ == '__main__':
	rospy.init_node('mapping')
	tf_ = TransformListener()
	transformer = TransformerROS()
	foo = mapping()
	rospy.spin()
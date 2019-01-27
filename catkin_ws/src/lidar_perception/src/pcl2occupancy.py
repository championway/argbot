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
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData
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
		self.pub_rviz = rospy.Publisher("/wp_line", Marker, queue_size = 1)
		self.resolution = 0.3
		self.width = 200
		self.height = 200
		self.origin = Pose()
		self.local_map = OccupancyGrid()
		self.dilating_size = 4
		self.start_planning = False
		self.goal = []
		self.astar = AStar()
		self.msg_count = 0
		self.border = 50
		self.frame_id = None

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
		self.msg_count = self.msg_count + 1
		self.init_param()
		self.frame_id = msg.header.frame_id
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

		if self.start_planning:
			self.path_planning()

	def path_planning(self):
		if self.msg_count % 5 != 0:
			return
		self.msg_count = 0
		cost_map = np.zeros((self.height, self.width))
		for i in range(self.height):
			for j in range(self.width):
				if i > int(self.border) and i < int(self.height - self.border):
					if j > int(self.border) and j < int(self.width - self.border):
						cost_map[i][j] = self.occupancygrid[i][j]
		start_point = self.map2occupancygrid((0, 0))
		start = (start_point[1], start_point[0])
		end = (self.goal[1], self.goal[0])
		self.astar.initial(cost_map, start, end)
		path = self.astar.planning()
		self.rviz(path)

	def cb_new_goal(self, p):
		self.goal = self.map2occupancygrid([p.pose.position.x, p.pose.position.y])
		self.start_planning = True

	def occupancygrid2map(self, p):
		x = p[0]*self.resolution + self.origin.position.x + self.resolution/2.
		y = p[1]*self.resolution + self.origin.position.y + self.resolution/2.
		return [x, y]

	def map2occupancygrid(self, p):
		x = int((p[0]-self.origin.position.x)/self.resolution)
		y = int((p[1]-self.origin.position.y)/self.resolution)
		return [x, y]
			
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
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
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from control.cfg import pos_PIDConfig, ang_PIDConfig, lookaheadConfig
from robotx_gazebo.msg import UsvDrive
from std_srvs.srv import SetBool, SetBoolResponse

from PID import PID_control
from pure_pursuit import PurePursuit

class Robot_PID():
	def __init__(self):
		self.node_name = rospy.get_name()
		self.dis4constV = 1. # Distance for constant velocity
		self.pos_ctrl_max = 0.5
		self.pos_ctrl_min = 0.0
		self.pos_station_max = 0.5
		self.pos_station_min = -0.5
		self.cmd_ctrl_max = 0.95
		self.cmd_ctrl_min = -0.95
		self.station_keeping_dis = 1

		self.is_station_keeping = False
		self.start_navigation = False
		self.stop_pos = []
		self.final_goal = None # The final goal that you want to arrive
		self.goal = self.final_goal
		self.robot_position = None

		rospy.loginfo("[%s] Initializing " %(self.node_name))

		self.sub_goal = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=1)
		rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb, queue_size = 1, buff_size = 2**24)
		self.pub_cmd = rospy.Publisher("/cmd_drive", UsvDrive, queue_size = 1)
		self.pub_lookahead = rospy.Publisher("/lookahead_point", Marker, queue_size = 1)
		self.station_keeping_srv = rospy.Service("/station_keeping", SetBool, self.station_keeping_cb)
		self.navigate_srv = rospy.Service("/navigation", SetBool, self.navigation_cb)

		self.pos_control = PID_control("Position")
		self.ang_control = PID_control("Angular")

		self.ang_station_control = PID_control("Angular_station")
		self.pos_station_control = PID_control("Position_station")

		self.purepursuit = PurePursuit()

		self.pos_srv = Server(pos_PIDConfig, self.pos_pid_cb, "Position")
		self.ang_srv = Server(ang_PIDConfig, self.ang_pid_cb, "Angular")
		self.pos_station_srv = Server(pos_PIDConfig, self.pos_station_pid_cb, "Angular_station")
		self.ang_station_srv = Server(ang_PIDConfig, self.ang_station_pid_cb, "Position_station")
		self.lookahead_srv = Server(lookaheadConfig, self.lookahead_cb, "LookAhead")

		self.initialize_PID()

	def odom_cb(self, msg):
		robot_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
		if not self.is_station_keeping:
			self.stop_pos = [[msg.pose.pose.position.x, msg.pose.pose.position.y]]
		quat = (msg.pose.pose.orientation.x,\
				msg.pose.pose.orientation.y,\
				msg.pose.pose.orientation.z,\
				msg.pose.pose.orientation.w)
		_, _, yaw = tf.transformations.euler_from_quaternion(quat)

		self.robot_position = robot_position
		if self.goal is None: # if the robot haven't recieve any goal
			return

		if not self.start_navigation:
			return
		reach_goal = self.purepursuit.set_robot_pose(robot_position, yaw)
		pursuit_point = self.purepursuit.get_pursuit_point()
		
		#yaw = yaw + np.pi/2.
		if reach_goal or reach_goal is None:
			self.publish_lookahead(robot_position, self.final_goal[-1])
			goal_distance = self.get_distance(robot_position, self.final_goal[-1])
			goal_angle = self.get_goal_angle(yaw, robot_position, self.final_goal[-1])
			pos_output, ang_output = self.station_keeping(goal_distance, goal_angle)
		elif self.is_station_keeping:
			rospy.loginfo("Station Keeping")
			self.publish_lookahead(robot_position, self.goal[0])
			goal_distance = self.get_distance(robot_position, self.goal[0])
			goal_angle = self.get_goal_angle(yaw, robot_position, self.goal[0])
			pos_output, ang_output = self.station_keeping(goal_distance, goal_angle)
		else:
			self.publish_lookahead(robot_position, pursuit_point)
			goal_distance = self.get_distance(robot_position, pursuit_point)
			goal_angle = self.get_goal_angle(yaw, robot_position, pursuit_point)
			pos_output, ang_output = self.control(goal_distance, goal_angle)
		
		cmd_msg = UsvDrive()
		cmd_msg.left = self.cmd_constarin(pos_output - ang_output)
		cmd_msg.right = self.cmd_constarin(pos_output + ang_output)
		self.pub_cmd.publish(cmd_msg)

	def control(self, goal_distance, goal_angle):
		self.pos_control.update(goal_distance)
		self.ang_control.update(goal_angle)

		# pos_output will always be positive
		pos_output = self.pos_constrain(-self.pos_control.output/self.dis4constV)

		# -1 = -180/180 < output/180 < 180/180 = 1
		ang_output = self.ang_control.output/180.
		return pos_output, ang_output

	def station_keeping(self, goal_distance, goal_angle):
		self.pos_station_control.update(goal_distance)
		self.ang_station_control.update(goal_angle)

		# pos_output will always be positive
		pos_output = self.pos_station_constrain(-self.pos_station_control.output/self.dis4constV)

		# -1 = -180/180 < output/180 < 180/180 = 1
		ang_output = self.ang_station_control.output/180.

		# if the goal is behind the robot
		if abs(goal_angle) > 90: 
			pos_output = - pos_output
			ang_output = - ang_output
		return pos_output, ang_output

	def goal_cb(self, p):
		if self.final_goal is None:
			self.final_goal = []
		self.final_goal.append([p.pose.position.x, p.pose.position.y])
		self.goal = self.final_goal

	def station_keeping_cb(self, req):
		if req.data == True:
			self.goal = self.stop_pos
			self.is_station_keeping = True
		else:
			self.is_station_keeping = False
		res = SetBoolResponse()
		res.success = True
		res.message = "recieved"
		return res

	def navigation_cb(self, req):
		if req.data == True:
			if not self.is_station_keeping:
				self.purepursuit.set_goal(self.robot_position, self.goal)
			self.is_station_keeping = False
			self.start_navigation = True
		else:
			self.start_navigation = False
			self.final_goal = None
			self.goal = self.stop_pos
		res = SetBoolResponse()
		res.success = True
		res.message = "recieved"
		return res

	def cmd_constarin(self, input):
		if input > self.cmd_ctrl_max:
			return self.cmd_ctrl_max
		if input < self.cmd_ctrl_min:
			return self.cmd_ctrl_min
		return input

	def pos_constrain(self, input):
		if input > self.pos_ctrl_max:
			return self.pos_ctrl_max
		if input < self.pos_ctrl_min:
			return self.pos_ctrl_min
		return input

	def pos_station_constrain(self, input):
		if input > self.pos_station_max:
			return self.pos_station_max
		if input < self.pos_station_min:
			return self.pos_station_min
		return input

	def initialize_PID(self):
		self.pos_control.setSampleTime(1)
		self.ang_control.setSampleTime(1)
		self.pos_station_control.setSampleTime(1)
		self.ang_station_control.setSampleTime(1)

		self.pos_control.SetPoint = 0.0
		self.ang_control.SetPoint = 0.0
		self.pos_station_control.SetPoint = 0.0
		self.ang_station_control.SetPoint = 0.0

	def get_goal_angle(self, robot_yaw, robot, goal):
		robot_angle = np.degrees(robot_yaw)
		p1 = [robot[0], robot[1]]
		p2 = [robot[0], robot[1]+1.]
		p3 = goal
		angle = self.get_angle(p1, p2, p3)
		result = angle - robot_angle
		result = self.angle_range(-(result + 90.))
		return result

	def get_angle(self, p1, p2, p3):
		v0 = np.array(p2) - np.array(p1)
		v1 = np.array(p3) - np.array(p1)
		angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
		return np.degrees(angle)

	def angle_range(self, angle): # limit the angle to the range of [-180, 180]
		if angle > 180:
			angle = angle - 360
			angle = self.angle_range(angle)
		elif angle < -180:
			angle = angle + 360
			angle = self.angle_range(angle)
		return angle

	def get_distance(self, p1, p2):
		return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

	def pos_pid_cb(self, config, level):
		print("Position: [Kp]: {Kp}   [Ki]: {Ki}   [Kd]: {Kd}\n".format(**config))
		Kp = float("{Kp}".format(**config))
		Ki = float("{Ki}".format(**config))
		Kd = float("{Kd}".format(**config))
		self.pos_control.setKp(Kp)
		self.pos_control.setKi(Ki)
		self.pos_control.setKd(Kd)
		return config

	def publish_lookahead(self, robot, lookahead):
		marker = Marker()
		marker.header.frame_id = "/odom"
		marker.header.stamp = rospy.Time.now()
		marker.ns = "pure_pursuit"
		marker.type = marker.LINE_STRIP
		marker.action = marker.ADD
		wp = Point()
		wp.x, wp.y = robot[:2]
		wp.z = 0
		marker.points.append(wp)
		wp = Point()
		wp.x, wp.y = lookahead[0], lookahead[1]
		wp.z = 0
		marker.points.append(wp)
		marker.id = 0
		marker.scale.x = 0.2
		marker.scale.y = 0.2
		marker.scale.z = 0.2
		marker.color.a = 1.0
		marker.color.b = 1.0
		marker.color.g = 1.0
		self.pub_lookahead.publish(marker)

	def ang_pid_cb(self, config, level):
		print("Angular: [Kp]: {Kp}   [Ki]: {Ki}   [Kd]: {Kd}\n".format(**config))
		Kp = float("{Kp}".format(**config))
		Ki = float("{Ki}".format(**config))
		Kd = float("{Kd}".format(**config))
		self.ang_control.setKp(Kp)
		self.ang_control.setKi(Ki)
		self.ang_control.setKd(Kd)
		return config

	def pos_station_pid_cb(self, config, level):
		print("Position: [Kp]: {Kp}   [Ki]: {Ki}   [Kd]: {Kd}\n".format(**config))
		Kp = float("{Kp}".format(**config))
		Ki = float("{Ki}".format(**config))
		Kd = float("{Kd}".format(**config))
		self.pos_station_control.setKp(Kp)
		self.pos_station_control.setKi(Ki)
		self.pos_station_control.setKd(Kd)
		return config

	def ang_station_pid_cb(self, config, level):
		print("Angular: [Kp]: {Kp}   [Ki]: {Ki}   [Kd]: {Kd}\n".format(**config))
		Kp = float("{Kp}".format(**config))
		Ki = float("{Ki}".format(**config))
		Kd = float("{Kd}".format(**config))
		self.ang_station_control.setKp(Kp)
		self.ang_station_control.setKi(Ki)
		self.ang_station_control.setKd(Kd)
		return config

	def lookahead_cb(self, config, level):
		print("Look Ahead Distance: {Look_Ahead}\n".format(**config))
		lh = float("{Look_Ahead}".format(**config))
		self.purepursuit.set_lookahead(lh)
		return config

if __name__ == '__main__':
	rospy.init_node('PID_control')
	foo = Robot_PID()
	rospy.spin()
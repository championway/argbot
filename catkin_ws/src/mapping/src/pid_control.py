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
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from mapping.cfg import pos_PIDConfig, ang_PIDConfig
from robotx_gazebo.msg import UsvDrive

class Robot_PID():
	def __init__(self):
		self.node_name = rospy.get_name()

		self.goal = [-10, -10]
		rospy.loginfo("[%s] Initializing " %(self.node_name))

		rospy.Subscriber('/odometry/filtered', Odometry, self.odom_cb, queue_size = 1, buff_size = 2**24)
		self.pub_cmd = rospy.Publisher("/cmd_drive", UsvDrive, queue_size = 1)

		self.pos_control = PID_control("Position")
		self.ang_control = PID_control("Angular")

		self.pos_srv = Server(pos_PIDConfig, self.pos_pid_cb, "Position")
		self.ang_srv = Server(ang_PIDConfig, self.ang_pid_cb, "Angular")

		self.initialize_PID()

	def odom_cb(self, msg):
		robot_position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
		quat = (msg.pose.pose.orientation.x,\
				msg.pose.pose.orientation.y,\
				msg.pose.pose.orientation.z,\
				msg.pose.pose.orientation.w)
		_, _, yaw = tf.transformations.euler_from_quaternion(quat)
		#yaw = yaw + np.pi/2
		goal_distance = self.get_distance(robot_position, self.goal)
		goal_angle = self.get_goal_angle(yaw, robot_position, self.goal)
		self.pos_control.update(goal_distance)
		self.ang_control.update(goal_angle)
		pos_output = self.pos_control.output
		ang_output = self.ang_control.output
		cmd_msg = UsvDrive()
		cmd_msg.left = ((pos_output+ang_output)/300. -1)
		cmd_msg.right = ((pos_output-ang_output)/300. -1)
		print(cmd_msg.left, cmd_msg.right)
		self.pub_cmd.publish(cmd_msg)

	def initialize_PID(self):
		self.pos_control.setSampleTime(1)
		self.ang_control.setSampleTime(1)

		self.pos_control.SetPoint = 0.0
		self.ang_control.SetPoint = 0.0

	def get_goal_angle(self, robot_yaw, robot, goal):
		robot_angle = np.degrees(robot_yaw)
		p1 = [robot[0], robot[1]]
		p2 = [robot[0], robot[1]+1.]
		p3 = goal
		angle = self.get_angle(p1, p2, p3)
		result = angle - robot_angle
		result = self.angle_range(result)
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

	def ang_pid_cb(self, config, level):
		print("Angular: [Kp]: {Kp}   [Ki]: {Ki}   [Kd]: {Kd}\n".format(**config))
		Kp = float("{Kp}".format(**config))
		Ki = float("{Ki}".format(**config))
		Kd = float("{Kd}".format(**config))
		self.ang_control.setKp(Kp)
		self.ang_control.setKi(Ki)
		self.ang_control.setKd(Kd)
		return config

class PID_control():
	def __init__(self, p_name, P=0.2, I=0.0, D=0.0):
		print "Turn on PID", p_name, "Control"
		self.Kp = P
		self.Ki = I
		self.Kd = D

		self.sample_time = 0.00
		self.current_time = rospy.get_time()
		self.last_time = self.current_time

		self.clear()

	def clear(self):
		"""Clears PID computations and coefficients"""
		self.SetPoint = 0.0

		self.PTerm = 0.0
		self.ITerm = 0.0
		self.DTerm = 0.0
		self.last_error = 0.0

		# Windup Guard
		self.int_error = 0.0
		self.windup_guard = 20.0

		self.output = 0.0


	def update(self, feedback_value):
		"""Calculates PID value for given reference feedback
		.. math::
			u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
		.. figure:: images/pid_1.png
			:align:   center
			Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
		"""
		error = self.SetPoint - feedback_value

		self.current_time = rospy.get_time()
		delta_time = self.current_time - self.last_time
		delta_error = error - self.last_error

		if (delta_time >= self.sample_time):
			self.PTerm = self.Kp * error
			self.ITerm += error * delta_time

			if (self.ITerm < -self.windup_guard):
				self.ITerm = -self.windup_guard
			elif (self.ITerm > self.windup_guard):
				self.ITerm = self.windup_guard

			self.DTerm = 0.0
			if delta_time > 0:
				self.DTerm = delta_error / delta_time

			# Remember last time and last error for next calculation
			self.last_time = self.current_time
			self.last_error = error

		self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

	def setKp(self, proportional_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
		self.Kp = proportional_gain

	def setKi(self, integral_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
		self.Ki = integral_gain

	def setKd(self, derivative_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
		self.Kd = derivative_gain

	def setWindup(self, windup):
		"""Integral windup, also known as integrator windup or reset windup,
		refers to the situation in a PID feedback controller where
		a large change in setpoint occurs (say a positive change)		a large change in setpoint occurs (say a positive change)
		and the integral terms accumulates a significant error
		during the rise (windup), thus overshooting and continuing
		to increase as this accumulated error is unwound
		(offset by errors in the other direction).
		The specific problem is the excess overshooting.
		"""
		self.windup_guard = windup

	def setSampleTime(self, sample_time):
		"""PID that should be updated at a regular interval.
		Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
		"""
		self.sample_time = sample_time

if __name__ == '__main__':
	rospy.init_node('PID_control')
	foo = Robot_PID()
	rospy.spin()
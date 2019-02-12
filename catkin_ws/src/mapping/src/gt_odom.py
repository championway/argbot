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
from gazebo_msgs.srv import GetModelState

class GT_odom():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.br = tf.TransformBroadcaster()
		self.pub_goal = rospy.Publisher('/odometry/ground_truth', Odometry, queue_size = 1)
		self.robot_pose = PoseStamped()
		self.robot_odom = Odometry()
		self.frame_id = "map"
		self.model_name = "X1"
		while True:
			self.show_gazebo_models()
			rospy.sleep(0.05)

	def show_gazebo_models(self):
		rospy.wait_for_service("/gazebo/get_model_state")
		try:
			model_pose = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
			resp = model_pose(self.model_name, "")
			self.robot_pose.header = resp.header
			self.robot_pose.header.frame_id = self.frame_id
			self.robot_pose.pose = resp.pose
			self.robot_odom.header = self.robot_pose.header
			self.robot_odom.pose.pose = self.robot_pose.pose
			self.pub_goal.publish(self.robot_odom)
			position = [resp.pose.position.x, resp.pose.position.y, resp.pose.position.z]
			quaternion = [resp.pose.orientation.x,\
					resp.pose.orientation.y,\
					resp.pose.orientation.z,\
					resp.pose.orientation.w]
			self.pub_tf(position, quaternion)
		except rospy.ServiceException as e:
			rospy.loginfo("Get Model State service call failed:  {0}".format(e))

	def pub_tf(self, position, quaternion):
		r, p, y = tf.transformations.euler_from_quaternion(quaternion)
		tfros = tf.TransformerROS()
		M = tfros.fromTranslationRotation(position, quaternion)
		M_inv =  np.linalg.inv(M)
		trans = tf.transformations.translation_from_matrix(M_inv)
		rot = tf.transformations.quaternion_from_matrix(M_inv)
		self.br.sendTransform(trans, rot, rospy.Time.now(), "map", "/X1/base_footprint")

if __name__ == '__main__':
	rospy.init_node('gt_odom')
	foo = GT_odom()
	rospy.spin()
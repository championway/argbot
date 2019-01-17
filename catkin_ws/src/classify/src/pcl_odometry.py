#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from robotx_msgs.msg import PCL_points, ObjectPose, ObjectPoseList
import rospkg
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError


class pcl_odometry():
	def __init__(self):
		self.br = tf.TransformBroadcaster()
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/pcl_points_img', PoseArray, self.call_back, queue_size = 1, buff_size = 2**24)
		self.pub_obj = rospy.Publisher("obj_list", ObjectPoseList, queue_size = 1)
		self.pub_marker = rospy.Publisher("/obj_classify", MarkerArray, queue_size = 1)
		self.pub_path = rospy.Publisher("/pcl/path", Path, queue_size = 1)
		#rospy.Subscriber('/pcl_array', PoseArray, self.call_back)
		self.height = self.width = 1400.
		self.scale = ((self.height-10.)/2.)/60.
		self.point_size = 1	# must be integer
		self.image = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		self.index = 0
		self.pre_image = None
		self.MAX_FEATURES = 600
		self.GOOD_MATCH_PERCENT = 0.15
		self.radius = 0.
		self.old_pos = [0, 0, 0]
		self.new_pos = [0, 0, 0]
		self.lidar_pose = PoseStamped()
		self.path = Path()

	def init_param(self):
		self.old_pos = self.new_pos
		self.pre_image = self.image.copy()

	def call_back(self, msg):
		self.init_param()
		pcl_point = PoseArray()
		pcl_point = msg
		#pcl_size = len(msg.poses)
		self.image = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		plane_xy = []
		plane_yz = []
		plane_xz = []
		pcl_size = len(pcl_point.poses)

		# ======= project to XY, YZ, XZ plane =======
		for j in range(pcl_size):
			plane_xy.append([pcl_point.poses[j].position.x, pcl_point.poses[j].position.y])
		self.toIMG(pcl_size, plane_xy)
		if self.pre_image is None:
			return
		M = self.alignImages(self.pre_image, self.image)
		#print M
		self.lidar2robot(M)
		self.drawPath()
		cv2.imwrite( "old.jpg", self.pre_image)
		cv2.imwrite( "new.jpg", self.image)
		self.br.sendTransform((self.lidar_pose.pose.position.x, \
			self.lidar_pose.pose.position.y, self.lidar_pose.pose.position.z), \
			(self.lidar_pose.pose.orientation.x, self.lidar_pose.pose.orientation.y, \
			self.lidar_pose.pose.orientation.z, self.lidar_pose.pose.orientation.w), \
			msg.header.stamp,"/velodyne","/odom")
		#cv2.imwrite( "Image.jpg", self.image)
		#cv2.imwrite( "Image_r_" + str(self.index) + ".jpg", self.image)
		#self.index = self.index + 1
		#print "Save image"
		#rospy.sleep(0.8)

	def lidar2robot(self, M):
		s = M[1][0]
		new_tran = [-M[0][2]/self.scale, -M[1][2]/self.scale]
		self.radius = (self.radius + np.arcsin(s)) % (2.*np.pi)
		s_ = np.sin(-self.radius)
		c_ = np.cos(-self.radius)
		quaternion = tf.transformations.quaternion_from_euler(0, 0, self.radius)
		rot_mat = np.array([[c_, -s_], [s_, c_]])
		new_pos_wo_rot = np.array(new_tran)
		new_pos = rot_mat.dot(new_pos_wo_rot)
		self.new_pos[:2] = [self.old_pos[0] + new_pos[0], self.old_pos[1] + new_pos[1]]
		self.lidar_pose.pose.position.x = self.new_pos[0]
		self.lidar_pose.pose.position.y = self.new_pos[1]
		self.lidar_pose.pose.position.z = 0
		self.lidar_pose.pose.orientation.x = quaternion[0]
		self.lidar_pose.pose.orientation.y = quaternion[1]
		self.lidar_pose.pose.orientation.z = quaternion[2]
		self.lidar_pose.pose.orientation.w = quaternion[3]
		print self.lidar_pose.pose.position
		self.lidar_pose.header.frame_id = "odom"
		return new_tran, c_, s_

	def drawPath(self):
		p = PoseStamped()
		#p.header = self.obj_list.header
		p.pose.position.x = self.new_pos[0]
		p.pose.position.y = self.new_pos[1]
		p.pose.position.z = 0
		self.path.poses.append(p)
		self.path.header.frame_id = "odom"
		self.pub_path.publish(self.path)

	def alignImages(self, im1Gray, im2Gray):
		# Convert images to grayscale
		#im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
		#im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

		# Detect ORB features and compute descriptors
		orb = cv2.ORB_create(self.MAX_FEATURES)
		keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
		keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

		# Match features
		matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
		matches = matcher.match(descriptors1, descriptors2, None)

		# Sort matches by score
		matches.sort(key=lambda x: x.distance, reverse=False)

		# Remove not so good matches
		numGoodMatches = int(len(matches) * self.GOOD_MATCH_PERCENT)
		matches = matches[:numGoodMatches]

		# Draw top matches
		imMatches = cv2.drawMatches(im1Gray, keypoints1, im2Gray, keypoints2, matches, None)
		cv2.imwrite("/home/joinet/pcl/matches.jpg", imMatches)
		# Extract location of good matches
		points1 = np.zeros((len(matches), 2), dtype=np.float32)
		points2 = np.zeros((len(matches), 2), dtype=np.float32)

		for i, match in enumerate(matches):
			points1[i, :] = keypoints1[match.queryIdx].pt
			points2[i, :] = keypoints2[match.trainIdx].pt

		# Find homography
		#h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
		
		# Find rigid transform
		M = cv2.estimateRigidTransform(points1, points2, False)

		return M

	def toIMG(self, pcl_size, pcl_array):
		scale = self.scale
		img = np.zeros((int(self.height), int(self.width), 3), np.uint8)
		for i in range(pcl_size):
			pcl_array[i][0] = int(round((pcl_array[i][0])*scale + self.height/2.))
			pcl_array[i][1] = int(round((pcl_array[i][1])*scale + self.width/2.))
			for m in range(-self.point_size, self.point_size + 1):
				for n in range(-self.point_size, self.point_size + 1):
					self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][0] = 255
					self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][1] = 255
					self.image[pcl_array[i][0] + m  , pcl_array[i][1] + n][2] = 255
		self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		#kernel = np.ones((5,5),np.float32)/25
		#self.image = cv2.filter2D(self.image, -1, kernel)
		#cv2.imwrite( "view.jpg", imGray)

if __name__ == '__main__':
	rospy.init_node('pcl_odometry')
	foo = pcl_odometry()
	rospy.spin()
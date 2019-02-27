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
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from control.cfg import pos_PIDConfig, ang_PIDConfig
from duckiepond_vehicle.msg import UsvDrive
from std_srvs.srv import SetBool, SetBoolResponse
from PID import PID_control
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from ssd import build_ssd
from matplotlib import pyplot as plt

class Diagnose():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.frame_id = 'odom'
		#self.image_sub = rospy.Subscriber("/BRIAN/camera_node/image/compressed", Image, self.img_cb, queue_size=1)
		self.image_sub = rospy.Subscriber("/BRIAN/camera_node/image/compressed", CompressedImage, self.img_cb, queue_size=1, buff_size = 2**24)
		self.pub_cmd = rospy.Publisher("/MONICA/cmd_drive", UsvDrive, queue_size = 1)
		self.pub_goal = rospy.Publisher("/goal_point", Marker, queue_size = 1)
		self.image_pub = rospy.Publisher("/predict_img", Image, queue_size = 1)
		self.station_keeping_srv = rospy.Service("/station_keeping", SetBool, self.station_keeping_cb)

		self.pos_control = PID_control("Position_tracking")
		self.ang_control = PID_control("Angular_tracking")

		self.ang_station_control = PID_control("Angular_station")
		self.pos_station_control = PID_control("Position_station")

		self.pos_srv = Server(pos_PIDConfig, self.pos_pid_cb, "Position_tracking")
		self.ang_srv = Server(ang_PIDConfig, self.ang_pid_cb, "Angular_tracking")
		self.pos_station_srv = Server(pos_PIDConfig, self.pos_station_pid_cb, "Angular_station")
		self.ang_station_srv = Server(ang_PIDConfig, self.ang_station_pid_cb, "Position_station")
		
		self.initialize_PID()

	def img_cb(self, msg):
		try:
			np_arr = np.fromstring(msg.data, np.uint8)
			cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
			#cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		(rows, cols, channels) = cv_image.shape
		self.width = cols
		self.height = rows
		predict = self.predict(cv_image)
		if predict is None:
			return
		angle, dis = predict[0], predict[1]
		self.tracking_control(angle, dis)


	def tracking_control(self, goal_angle, goal_distance):
		if self.is_station_keeping:
			rospy.loginfo("Station Keeping")
			pos_output, ang_output = self.station_keeping(goal_distance, goal_angle)
		else:
			pos_output, ang_output = self.control(goal_distance, goal_angle)
		cmd_msg = UsvDrive()
		cmd_msg.left = self.cmd_constarin(pos_output + ang_output)
		cmd_msg.right = self.cmd_constarin(pos_output - ang_output)
		self.pub_cmd.publish(cmd_msg)
		#self.publish_goal(self.goal)

	def predict(self, img):
		# Image Preprocessing (vgg use BGR image as training input)
		image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		x = cv2.resize(image, (300, 300)).astype(np.float32)
		x -= (104.0, 117.0, 123.0)
		x = x.astype(np.float32)
		x = x[:, :, ::-1].copy()
		x = torch.from_numpy(x).permute(2, 0, 1)

		#SSD Prediction
		xx = Variable(x.unsqueeze(0))
		if torch.cuda.is_available():
			xx = xx.cuda()
		y = self.net(xx)
		scale = torch.Tensor(img.shape[1::-1]).repeat(2)
		detections = y.data
		max_prob = 0
		coords = None
		for i in range(self.ROBOT_NUM):
			if detections[0, 1, i, 0].numpy() > self.predict_prob and detections[0, 1, i, 0].numpy() > max_prob:
				max_prob = detections[0, 1, i, 0].numpy()
				score = detections[0, 1, i, 0]
				pt = (detections[0, 1, i,1:]*scale).cpu().numpy()
				coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
		if coords is None:
			return None
		angle, dis, center = self.BBx2AngDis(coords)
		cv2.circle(img, (int(center[0]), int(center[1])), 10, (0,0,255), -1)
		cv2.rectangle(img, (int(coords[0][0]), int(coords[0][1])),\
							(int(coords[0][0] + coords[1]), int(coords[0][1] + coords[2])),(0,0,255),5)
		try:
			img = self.draw_cmd(img, dis, angle)
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
		except CvBridgeError as e:
			print(e)

if __name__ == '__main__':
	rospy.init_node('diagnose')
	foo = Diagnose()
	rospy.spin()
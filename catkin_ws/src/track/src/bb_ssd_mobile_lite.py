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
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import rospkg
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from bb_pointnet.msg import * 
import os 
import message_filters

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
#from vision.utils.misc import Timer

class bb_ssd_mobile_lite(object):
	def __init__(self):
		model = "v1"
		self.prob_threshold = 0.85
		self.cv_bridge = CvBridge() 
		self.num_points = 8000
		self.labels = ['background' , # always index 0
				'person','palm']
		self.objects = []
		if model == "v2_lite":
			self.network = create_mobilenetv2_ssd_lite(len(self.labels), is_test=True) 
		elif model == "v1":
			self.network = create_mobilenetv1_ssd(len(self.labels), is_test=True) 
		elif model == "v1_lite":
			self.network = create_mobilenetv1_ssd_lite(len(self.labels), is_test=True) 

		model_path = '/home/arg_ws3/pytorch-ssd/models/argbot_person_palm/mb1-ssd-Epoch-10-Loss-3.1767.pth'
		state_dict = torch.load(os.path.join(model_path))
		self.network.load_state_dict(state_dict)
		DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.network.to(DEVICE)
		if model == "v2_lite":
			self.predictor = create_mobilenetv2_ssd_lite_predictor(self.network, candidate_size=200, device = DEVICE)
		elif model == "v1_lite":
			self.predictor = create_mobilenetv1_ssd_lite_predictor(self.network, candidate_size=200, device = DEVICE)
		elif model == "v1":	
			self.predictor = create_mobilenetv1_ssd_predictor(self.network, candidate_size=200, device = DEVICE)

		#### Publisher
		self.origin = rospy.Publisher('/input', bb_input, queue_size=1)
		self.image_pub = rospy.Publisher("/predict_img", Image, queue_size = 1)
		self.mask_pub = rospy.Publisher("/predict_mask", Image, queue_size = 1)

		### msg filter 
		self.is_compressed = False

		video_mode = False 
		if video_mode:
			image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.video_callback)
		else:
			image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
			depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
			ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
			ts.registerCallback(self.callback)

	def callback(self, img_msg, depth):
		try:
			if self.is_compressed:
				np_arr = np.fromstring(img_msg.data, np.uint8)
				cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
			else:
				cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		img = cv_image.copy()
		

		(rows, cols, channels) = cv_image.shape
		self.width = cols
		self.height = rows
		#predict_img, obj_list = self.predict(cv_image)
		predict_img = self.predict(cv_image)
		try:
			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict_img, "bgr8"))
		except CvBridgeError as e:
			print(e)

		# for obj in obj_list:
		# 	out = bb_input()
		# 	obj[0] = obj[0] - 5
		# 	obj[1] = obj[1] - 5
		# 	obj[2] = obj[2] + 10
		# 	obj[3] = obj[3] + 10

		# 	mask = np.zeros((rows, cols), dtype = np.uint8)
		# 	point_list = [(int(obj[0]), int(obj[1])),(int(obj[0] + obj[2]),int(obj[1])),\
		# 		(int(obj[0] + obj[2]), int(obj[1] + obj[3])), (int(obj[0]), int(obj[1] + obj[3]))]

		# 	cv2.fillConvexPoly(mask, np.asarray(point_list,dtype = np.int), 255)
		# 	# print point_list
		# 	out.image = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")
		# 	out.mask = self.cv_bridge.cv2_to_imgmsg(mask, "8UC1")
		# 	out.depth = depth
		# 	out.header = img_msg.header
		# 	self.mask_pub.publish(out.mask)
		# 	self.origin.publish(out)

		# try:
		# 	img = self.cv_bridge.imgmsg_to_cv2(msg.data, "bgr8")
		# 	depth = self.cv_bridge.imgmsg_to_cv2(msg.depth, "16FC1")
		# 	mask = self.cv_bridge.imgmsg_to_cv2(msg.mask, "64FC1")
		# except CvBridgeError as e:
		# 	print(e)
	def video_callback(self, img_msg):

		try:
			if self.is_compressed:
				np_arr = np.fromstring(img_msg.data, np.uint8)
				cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
			else:
				cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		img = cv_image.copy()
		

		(rows, cols, channels) = cv_image.shape
		self.width = cols
		self.height = rows
		#predict_img, obj_list = self.predict(cv_image)
		predict_img = self.predict(cv_image)
		try:
			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict_img, "bgr8"))
		except CvBridgeError as e:
			print(e)

		# for obj in obj_list:
		# 	mask = np.zeros((rows, cols), dtype = np.uint8)
		# 	point_list = [(int(obj[0]), int(obj[1])),(int(obj[0] + obj[2]),int(obj[1])),\
		# 		(int(obj[0] + obj[2]), int(obj[1] + obj[3])), (int(obj[0]), int(obj[1] + obj[3]))]

		# 	cv2.fillConvexPoly(mask, np.asarray(point_list,dtype = np.int), 255)
		# 	mask = self.cv_bridge.cv2_to_imgmsg(mask, "8UC1")
		# 	self.mask_pub.publish(mask)

	def predict(self, img):
		# Preprocessing

		image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		time = rospy.get_time()
		boxes, labels, probs = self.predictor.predict(image, 10, self.prob_threshold)
		print(1./(rospy.get_time() - time))
		for i in range(boxes.size(0)):
			box = boxes[i, :]
			if (box[0], box[1]) != (box[2], box[3]):
				cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

				label = "{}: {:.2f}".format(self.labels[labels[i]], probs[i])
				cv2.putText(img, label,(box[0] + 20, box[1] + 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)


		# for obj in objs:
		# 	if obj[4] == 1:
		# 		color = (0, 255, 255)
		# 	elif obj[4] == 2:
		# 		color = (255, 255, 0)
		# 	elif obj[4] == 3:
		# 		color = (255, 0, 255)
		# 	else:
		# 		color = (0, 0, 0)
		# 	cv2.rectangle(img, (int(obj[0]), int(obj[1])),\
		# 						(int(obj[0] + obj[2]), int(obj[1] + obj[3])), color, 3)
		# 	cv2.putText(img, self.labels[obj[4]], (int(obj[0] + obj[2]), int(obj[1])), 0, 1, color,2)

		return img# , objs


	def onShutdown(self):
		rospy.loginfo("Shutdown.")


if __name__ == '__main__': 
	rospy.init_node('bb_ssd_mobile_lite',anonymous=False)
	bb_ssd_mobile_lite = bb_ssd_mobile_lite()
	rospy.on_shutdown(bb_ssd_mobile_lite.onShutdown)
	rospy.spin()

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
from ssd import build_ssd
import os 
import message_filters

class bb_ssd_prediction(object):
	def __init__(self):
		self.prob_threshold = 0.1
		self.cv_bridge = CvBridge() 
		self.num_points = 8000
		self.labels = ['background' , # always index 0
				'bb_extinguisher','bb_drill','bb_backpack']
		self.objects = []
		self.network = build_ssd('test', 300, 4) 
		self.is_compressed = False

		self.cuda_use = torch.cuda.is_available()
		#self.cuda_use = False

		if self.cuda_use:
			self.network = self.network.cuda()
		model_dir = "/home/andyser/code/subt_related/subt_arti_searching/ssd/weights"
		model_name = "ssd300_subt_110000.pth"	
		state_dict = torch.load(os.path.join('/home/arg_ws3/argbot/catkin_ws/src/classify/models/ssd300_subt_40000.pth'))
		self.network.load_state_dict(state_dict)
		#### Publisher
		self.origin = rospy.Publisher('/input', bb_input, queue_size=1)
		self.image_pub = rospy.Publisher("/predict_img", Image, queue_size = 1)
		self.mask_pub = rospy.Publisher("/predict_mask", Image, queue_size = 1)

		### msg filter 

		video_mode = True 
		if video_mode:
			image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.video_callback)
		else:
			image_sub = message_filters.Subscriber('/camera/color/image_rect_color', Image)
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
		predict_img, obj_list = self.predict(cv_image)
		try:
			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict_img, "bgr8"))
		except CvBridgeError as e:
			print(e)

		for obj in obj_list:
			out = bb_input()
			# obj[0] = obj[0] - 30
			# obj[1] = obj[1]	- 30
			# obj[2] = obj[2] + 100
			# obj[3] = obj[3] + 100

			mask = np.zeros((rows, cols), dtype = np.uint8)
			point_list = [(int(obj[0]), int(obj[1])),(int(obj[0] + obj[2]),int(obj[1])),\
				(int(obj[0] + obj[2]), int(obj[1] + obj[3])), (int(obj[0]), int(obj[1] + obj[3]))]

			cv2.fillConvexPoly(mask, np.asarray(point_list,dtype = np.int), 255)
			# print point_list
			out.image = self.cv_bridge.cv2_to_imgmsg(img, "bgr8")
			out.mask = self.cv_bridge.cv2_to_imgmsg(mask, "8UC1")
			out.depth = depth
			out.header = img_msg.header
			self.mask_pub.publish(out.mask)
			self.origin.publish(out)

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
		predict_img, obj_list = self.predict(cv_image)
		try:
			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict_img, "bgr8"))
		except CvBridgeError as e:
			print(e)

		for obj in obj_list:
			mask = np.zeros((rows, cols), dtype = np.uint8)
			point_list = [(int(obj[0]), int(obj[1])),(int(obj[0] + obj[2]),int(obj[1])),\
				(int(obj[0] + obj[2]), int(obj[1] + obj[3])), (int(obj[0]), int(obj[1] + obj[3]))]

			cv2.fillConvexPoly(mask, np.asarray(point_list,dtype = np.int), 255)
			mask = self.cv_bridge.cv2_to_imgmsg(mask, "8UC1")
			self.mask_pub.publish(mask)
	def predict(self, img):
		# Preprocessing
		image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		x = cv2.resize(image, (300, 300)).astype(np.float32)
		x -= (104.0, 117.0, 123.0)
		x = x.astype(np.float32)
		x = x[:, :, ::-1].copy()
		x = torch.from_numpy(x).permute(2, 0, 1)

		#SSD Forward Pass
		xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
		if self.cuda_use:
			xx = xx.cuda()
		y = self.network(xx)
		scale = torch.Tensor(img.shape[1::-1]).repeat(2)
		detections = y.data	# torch.Size([1, 4, 200, 5]) --> [batch?, class, object, coordinates]
		objs = []
		for i in range(detections.size(1)): # detections.size(1) --> class size
			for j in range(5):	# each class choose top 5 predictions
				if detections[0, i, j, 0].numpy() > self.prob_threshold:
					score = detections[0, i, j, 0]
					pt = (detections[0, i, j,1:]*scale).cpu().numpy()
					objs.append([pt[0], pt[1], pt[2]-pt[0]+1, pt[3]-pt[1]+1, i])

		for obj in objs:
			if obj[4] == 0:
				color = (0, 255, 255)
			elif obj[4] == 1:
				color = (255, 255, 0)
			elif obj[4] == 2:
				color = (255, 0, 255)
			else:
				color = (0, 0, 0)
			cv2.rectangle(img, (int(obj[0]), int(obj[1])),\
								(int(obj[0] + obj[2]), int(obj[1] + obj[3])), color, 3)
			cv2.putText(img, self.labels[obj[4]], (int(obj[0] + obj[2]), int(obj[1])), 0, 1, color,2)
			#print(self.labels[obj[4]])
		#cv2.imshow('image',img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
		return img, objs


	def onShutdown(self):
		rospy.loginfo("Shutdown.")	


	def getXYZ(self,xp, yp, zc):
		#### Definition:
		# cx, cy : image center(pixel)
		# fx, fy : focal length
		# xp, yp: index of the depth image
		# zc: depth
		inv_fx = 1.0/self.fx
		inv_fy = 1.0/self.fy
		x = (xp-self.cx) *  zc * inv_fx
		y = (yp-self.cy) *  zc * inv_fy
		z = zc
		return (x,y,z)			


if __name__ == '__main__': 
	rospy.init_node('bb_ssd_prediction',anonymous=False)
	bb_ssd_prediction = bb_ssd_prediction()
	rospy.on_shutdown(bb_ssd_prediction.onShutdown)
	rospy.spin()

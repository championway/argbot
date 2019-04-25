#!/usr/bin/env python

import Tracker
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

class PERSON_TRACKING():
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

		model_path = '/home/arg_ws3/pytorch-ssd/models/argbot_person_palm/mb1-ssd-Epoch-50-Loss-2.5815.pth'
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

		self.x_np = np.linspace(0, 40, 40, dtype=np.double)
		self.y_np = np.sin(np.linspace(0, 40, 40, dtype=np.double))
		self.tracker = Tracker.Tracker()

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
		predict_img = self.predict(cv_image)
		try:
			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict_img, "bgr8"))
		except CvBridgeError as e:
			print(e)

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
		predict_img = self.predict(cv_image)
		try:
			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict_img, "bgr8"))
		except CvBridgeError as e:
			print(e)

	def predict(self, img):
		image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		time = rospy.get_time()
		boxes, labels, probs = self.predictor.predict(image, 10, self.prob_threshold)
		print(1./(rospy.get_time() - time))
		for i in range(boxes.size(0)):
			box = boxes[i, :]
			if (box[0], box[1]) != (box[2], box[3]):
				bbx = [box[0], box[1], box[2], box[3]]
				cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
				cv2.circle(img, self.get_center(bbx), 5, (0,0,255), -1)

				label = "{}: {:.2f}".format(self.labels[labels[i]], probs[i])
				cv2.putText(img, label,(box[0] + 20, box[1] + 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)
		return img

	def get_center(self, bbx):
		center = ((bbx[2]+bbx[0])/2., (bbx[3]+bbx[1])/2.)
		return center

	def get_iou(self, bbx1, bbx2): # bbx = [x1, y1, x2, y2]
		area1 = abs((bbx1[0] - bbx1[2])*(bbx1[1] - bbx1[3]))
		area2 = abs((bbx2[0] - bbx2[2])*(bbx2[1] - bbx2[3]))
		dx = min(bbx1[2], bbx2[2]) - max(bbx1[0], bbx2[0])
		dy = min(bbx1[3], bbx2[3]) - max(bbx1[1], bbx2[1])
		if (dx >= 0) and (dy >= 0):
			return dx*dy/min(area1, area2)

	def tracking(self):
		print(self.tracker.get_prediction())
		self.tracker.Update([2,2])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)
		self.tracker.Update([2,2])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)
		self.tracker.Update([2,2])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)
		self.tracker.Update([2,2])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)
		self.tracker.Update([2,2])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)
		self.tracker.Update([2,2])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)
		self.tracker.Update([5,4])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)
		self.tracker.Update([9,3])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)
		self.tracker.Update([2,9])
		print(self.tracker.get_prediction())
		print(self.tracker.track.KF.P)

	def onShutdown(self):
		rospy.loginfo("Shutdown.")

if __name__=='__main__':
	rospy.init_node('PERSON_TRACKING')
	foo = PERSON_TRACKING()
	rospy.spin()
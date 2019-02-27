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

class SUBT_detection():
	def __init__(self):
		#self.node_name = rospy.get_name()
		#rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.bridge = CvBridge()
		self.is_compressed = True
		# Image definition
		self.width = 640
		self.height = 480
		self.labels = ['background', 'people', 'palm']
		self.prob_threshold = 0.9
		self.objects = []
		self.net = build_ssd('test', 300, len(self.labels))    # initialize SSD
		self.net.load_weights('/home/arg_ws3/ssd.pytorch/weights/argbot/argbot_54000.pth')
		if torch.cuda.is_available():
			self.net = self.net.cuda()

		#img = cv2.imread("radio.jpg")
		self.image_pub = rospy.Publisher("/predict_img", Image, queue_size = 1)
		self.image_sub = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.img_cb, queue_size=1, buff_size = 2**24)
		#self.predict(img)

	def img_cb(self, msg):
		try:
			if self.is_compressed:
				np_arr = np.fromstring(msg.data, np.uint8)
				cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
			else:
				cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
		except CvBridgeError as e:
			print(e)
		(rows, cols, channels) = cv_image.shape
		self.width = cols
		self.height = rows
		predict_img = self.predict(cv_image)
		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(predict_img, "bgr8"))
		except CvBridgeError as e:
			print(e)

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
		if torch.cuda.is_available():
			xx = xx.cuda()
		y = self.net(xx)
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
		return img


if __name__ == '__main__':
	rospy.init_node('subt_detection')
	foo = SUBT_detection()
	rospy.spin()
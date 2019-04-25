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
import Tracker

class PERSON_detection():
	def __init__(self):
		#self.node_name = rospy.get_name()
		#rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.bridge = CvBridge()
		self.is_compressed = False
		# Image definition
		self.width = 640
		self.height = 480
		self.labels = ['background', 'person', 'palm']
		self.prob_threshold = 0.3
		self.objects = []
		self.net = build_ssd('test', 300, len(self.labels))    # initialize SSD
		self.net.load_weights('/home/user/argbot/catkin_ws/src/track/src/argbot_58000.pth')
		if torch.cuda.is_available():
			self.net = self.net.cuda()

		#img = cv2.imread("radio.jpg")
		self.image_pub = rospy.Publisher("/predict_img", Image, queue_size = 1)
		self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb, queue_size=1, buff_size = 2**24)
		self.tracker = Tracker.Tracker()
		self.path = []
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
		cv2.imwrite('test.jpg', cv_image)
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
		t1 = rospy.get_time()
		y = self.net(xx)
		t2 = rospy.get_time()
		print(1./(t2-t1))
		scale = torch.Tensor(img.shape[1::-1]).repeat(2)
		detections = y.data	# torch.Size([1, 4, 200, 5]) --> [batch?, class, object, coordinates]
		objs = []
		for i in range(detections.size(1)): # detections.size(1) --> class size
			for j in range(5):	# each class choose top 5 predictions
				if detections[0, i, j, 0].numpy() > self.prob_threshold:
					# print(detections[0, i, j, 0].numpy())
					score = detections[0, i, j, 0]
					pt = (detections[0, i, j,1:]*scale).cpu().numpy()
					objs.append([pt[0], pt[1], pt[2]-pt[0]+1, pt[3]-pt[1]+1, i])
		palm_cnt = 0
		for obj in objs:
			if obj[4] == 0:
				color = (0, 255, 255)
			elif obj[4] == 1:
				color = (255, 255, 0)
			elif obj[4] == 2:
				color = (255, 0, 255)
			else:
				color = (0, 0, 0)
			bbx = [obj[0], obj[1], obj[0] + obj[2], obj[1] + obj[3]]
			cv2.rectangle(img, (int(obj[0]), int(obj[1])),\
								(int(obj[0] + obj[2]), int(obj[1] + obj[3])), color, 3)
			center = self.get_center(bbx)
			cv2.circle(img, center, 5, (0,0,255), -1)
			# cv2.putText(img, self.labels[obj[4]], (int(obj[0] + obj[2]), int(obj[1])), 0, 1, color,2)
			if self.labels[obj[4]] == 'palm':
				palm_cnt = palm_cnt + 1
				self.path.append(list(center))
		if palm_cnt > 1:
			self.path = []
		draw_path = np.array(self.path, np.int32)
		draw_path = draw_path.reshape((-1,1,2))
		cv2.polylines(img, [draw_path], False, (0,255,255))
		return img

	def get_center(self, bbx):
		center = (int((bbx[2]+bbx[0])/2.), int((bbx[3]+bbx[1])/2.))
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


if __name__ == '__main__':
	rospy.init_node('PERSON_detection')
	foo = PERSON_detection()
	rospy.spin()
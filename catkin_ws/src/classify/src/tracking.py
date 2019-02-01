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
import rospkg
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from ssd import build_ssd
from matplotlib import pyplot as plt
class pcl_odometry():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.ROBOT_NUM = 3
		# Image definition
		self.width = 1280
		self.height = 960
		self.wavm_labels = ["wamv",""]
		#rospy.Subscriber('/pcl_points_img', PoseArray, self.call_back, queue_size = 1, buff_size = 2**24)
		self.net = build_ssd('test', 300, 3)    # initialize SSD
		self.net.load_weights('/home/arg_ws3/argbot/catkin_ws/src/classify/src/ssd300_wamv_5000.pth')
		if torch.cuda.is_available():
			self.net = self.net.cuda()
		img = cv2.imread('miniwamv0545.jpg')
		self.predict(img)

	def predict(self, img):
		# Preprocessing
		image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		x = cv2.resize(image, (300, 300)).astype(np.float32)
		x -= (104.0, 117.0, 123.0)
		x = x.astype(np.float32)
		x = x[:, :, ::-1].copy()
		x = torch.from_numpy(x).permute(2, 0, 1)

		plt.figure(figsize=(10,10))
		

		#SSD Forward Pass
		xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
		if torch.cuda.is_available():
			xx = xx.cuda()
		y = self.net(xx)
		scale = torch.Tensor(img.shape[1::-1]).repeat(2)
		detections = y.data
		currentAxis = plt.gca()
		coords = None
		for i in range(self.ROBOT_NUM):
			if detections[0, 1, i, 0].numpy() > 0.6:
				score = detections[0, 1, i, 0]
				pt = (detections[0, 1, i,1:]*scale).cpu().numpy()
				coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
				print(coords)
		angle, dis, center = self.BBx2AngDis(coords)
		print(img.shape)
		cv2.circle(img, (int(center[0]), int(center[1])), 10, (0,0,255), -1)
		cv2.rectangle(img, (int(coords[0][0]), int(coords[0][1])),\
							(int(coords[0][0] + coords[1]), int(coords[0][1] + coords[2])),(0,0,255),3)
		#cv2.rectangle(img, (480, 480),(10, 20),(0,255,0),3)
		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		'''for i in range(self.ROBOT_NUM):
			print(image.shape)
			if detections[0, 1, i, 0].numpy() > 0.6:
				print(detections[0, 1, i, 0])
				score = detections[0, 1, i, 0]
				pt = (detections[0, 1, i,1:]*scale).cpu().numpy()
				coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
				print(coords)
				center = self.bbx2control(coords)
				color = colors[i]
				display_txt = '%s: %.2f'%('wawmv', score)
				currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
				currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
				circle1 = plt.Circle((center[0], center[1]), 10, color='r')
				currentAxis.add_artist(circle1)
		plt.show()'''

	def BBx2AngDis(self, coords):
		x = coords[0][0]
		y = coords[0][1]
		w = coords[1]
		h = coords[2]
		center = (x + w/2., y + h/2.)
		angle = self.width/2. - center[0]
		dis = (h*10. + w)/(self.height*10. + self.width)
		return angle, dis, center

	def init_param(self):
		pass

if __name__ == '__main__':
	rospy.init_node('pcl_odometry')
	foo = pcl_odometry()
	#rospy.spin()
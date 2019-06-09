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
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs.msg import CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, PoseStamped, Point
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

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

class PERSON_TRACKING():
	def __init__(self):

		model = "v1"
		self.prob_threshold = 0.85
		self.cv_bridge = CvBridge() 
		self.labels = ['background' , # always index 0
				'person','palm']
		self.objects = []
		if model == "v2_lite":
			self.network = create_mobilenetv2_ssd_lite(len(self.labels), is_test=True) 
		elif model == "v1":
			self.network = create_mobilenetv1_ssd(len(self.labels), is_test=True) 
		elif model == "v1_lite":
			self.network = create_mobilenetv1_ssd_lite(len(self.labels), is_test=True) 

		model_path = '/home/arg_ws3/pytorch-ssd/models/argbot_person_palm_new_new/mb1-ssd-Epoch-749-Loss-1.8576.pth'
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
		self.pub_tracking = rospy.Publisher("/tracking_point", Marker, queue_size = 1)
		self.pub_point_array = rospy.Publisher("/person_point_array", MarkerArray, queue_size = 1)

		self.kf_x = KalmanFilter(dim_x=2, dim_z=1)
		self.kf_y = KalmanFilter(dim_x=2, dim_z=1)
		self.path = []
		self.person_list = []
		self.cv_depthimage = None
		self.start_tracking = False
		self.frame_id = 'camera_link'
		self.t_old = None 
		self.t_now = None

		info_msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=None)
		self.fx = info_msg.P[0]
		self.fy = info_msg.P[5]
		self.cx = info_msg.P[2]
		self.cy = info_msg.P[6]

		### msg filter 
		self.is_compressed = False

		image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
		depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
		ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
		ts.registerCallback(self.callback)

	def stamp2time(self, stamp):
		return stamp.secs + stamp.nsecs*10e-9

	def init_kf(self, pos):
		self.kf_x.x = np.array([[pos[0]], [0.]])
		self.kf_x.F = np.array([[1., 1.], [0., 1.]])
		self.kf_x.H = np.array([[1., 0.]])
		self.kf_x.P = np.array([[1000., 0.], [0., 1000.]])
		self.kf_x.R = 5
		self.kf_x.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

		self.kf_y.x = np.array([[pos[1]], [0.]])
		self.kf_y.F = np.array([[1., 1.], [0., 1.]])
		self.kf_y.H = np.array([[1., 0.]])
		self.kf_y.P = np.array([[1000., 0.], [0., 1000.]])
		self.kf_y.R = 5
		self.kf_y.Q = Q_discrete_white_noise(dim=2, dt=0.1, var=0.13)

	def callback(self, img_msg, depth_msg):
		try:
			if self.is_compressed:
				np_arr = np.fromstring(img_msg.data, np.uint8)
				cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
				self.cv_depthimage = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
			else:
				cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")
				self.cv_depthimage = self.cv_bridge.imgmsg_to_cv2(depth_msg, "16UC1")
		except CvBridgeError as e:
			print(e)
		img = cv_image.copy()
		if self.t_old is None:
			self.t_old = self.stamp2time(img_msg.header.stamp)
		self.t_now = self.stamp2time(img_msg.header.stamp)
		

		(rows, cols, channels) = cv_image.shape
		self.width = cols
		self.height = rows
		predict_img = self.predict(cv_image)
		try:
			self.image_pub.publish(self.cv_bridge.cv2_to_imgmsg(predict_img, "bgr8"))
		except CvBridgeError as e:
			print(e)

	def predict(self, img):
		self.person_list = []
		coordinate_list = []
		image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		time = rospy.get_time()
		boxes, labels, probs = self.predictor.predict(image, 10, self.prob_threshold)
		#print(1./(rospy.get_time() - time))
		for i in range(boxes.size(0)):
			box = boxes[i, :]
			if (box[0], box[1]) != (box[2], box[3]):
				bbx = [box[0], box[1], box[2], box[3]]
				center = self.get_center(bbx)
				cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
				cv2.circle(img, center, 5, (0,0,255), -1)
				label = self.labels[labels[i]]
				cv2_text = "{}: {:.2f}".format(label, probs[i])
				cv2.putText(img, cv2_text,(box[0] + 20, box[1] + 40),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 255),2)
				if label == 'person':
					c_depth = self.center_depth(center)
					coordinate = self.getXYZ(center[0], center[1], c_depth)
					if coordinate != (0., 0., 0.):
						self.person_list.append(coordinate)
						#self.draw_point(coordinate)
		if len(self.person_list) > 0:
			if not self.start_tracking:
				self.start_tracking = True
				self.init_kf(self.person_list[0])
			else:
				self.tracking()
				self.draw_point_array(self.person_list)
				self.draw_tracking([self.kf_x.x[0], self.kf_y.y[0]])
				self.t_old = self.t_now	
		return img

	def tracking(self):
		dt = self.t_now - self.t_old

		self.kf_x.F = np.array([[1., dt], [0., 1.]])
		self.kf_x.predict()

		self.kf_y.F = np.array([[1., dt], [0., 1.]])
		self.kf_y.predict()

		predict_x = self.kf_x.x[0]
		predict_y = self.kf_y.x[0]

		dis_list = []
		for person in self.person_list:
			dis = self.distance(person, [predict_x, predict_y])
			dis_list.append(dis)
		dis_list = np.array(dis_list)
		min_dis = dis_list.min()
		goal_idx, = np.where(dis_list == min_dis)[0]

		uncertainty = min_dis**2
		self.kf_x.R = uncertainty
		self.kf_y.R = uncertainty
		self.kf_x.update([self.person_list[goal_idx][0]])
		self.kf_y.update([self.person_list[goal_idx][1]])

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

	def center_depth(self, center):
		self.cv_depthimage = self.cv_depthimage/1000.
		cnt_size = 3
		depth_list = []
		for i in range(-cnt_size, cnt_size):
			for j in range(-cnt_size, cnt_size):
				depth_list.append(self.cv_depthimage[center[1] + i, center[0] - j])
		depth_list.sort()
		return depth_list[int(len(depth_list)/2)]

	def get_center(self, bbx):
		center = (int((bbx[2]+bbx[0])/2.), int((bbx[3]+bbx[1])/2.))
		return center

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
	    return (z, -x, -y)

	def distance(self, a, b):
		return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

	def draw_point(self, coordinate):
		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.type = marker.POINTS
		marker.pose.orientation.w = 1
		marker.scale.x = 0.3
		marker.scale.y = 0.3
		marker.scale.z = 0.3
		marker.color.a = 0.7
		marker.color.g = 1.0
		p = Point()
		p.x = coordinate[0]
		p.y = coordinate[1]
		marker.points = [p]
		self.pub_point.publish(marker)

	def draw_point_array(self, coordinate):
		marker_array = MarkerArray()
		idx = 1
		for coor in coordinate:
			marker = Marker()
			marker.header.frame_id = self.frame_id
			marker.type = marker.CUBE
			marker.action = marker.ADD
			marker.pose.orientation.w = 1
			marker.scale.x = 0.2
			marker.scale.y = 0.2
			marker.scale.z = 0.2
			marker.color.a = 0.7
			marker.color.b = 1.0
			#marker.id = idx
			#idx = idx + 1
			marker.pose.position.x = coor[0]
			marker.pose.position.y = coor[1]
			marker_array.markers.append(marker)
		self.pub_point_array.publish(marker_array)

	def draw_tracking(self, coordinate):
		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.type = marker.POINTS
		marker.pose.orientation.w = 1
		marker.scale.x = 0.2
		marker.scale.y = 0.2
		marker.scale.z = 0.2
		marker.color.a = 0.7
		marker.color.r = 1.0
		p = Point()
		p.x = coordinate[0]
		p.y = coordinate[1]
		marker.points = [p]
		self.pub_tracking.publish(marker)

	def onShutdown(self):
		rospy.loginfo("Shutdown.")

if __name__=='__main__':
	rospy.init_node('PERSON_TRACKING')
	foo = PERSON_TRACKING()
	rospy.spin()
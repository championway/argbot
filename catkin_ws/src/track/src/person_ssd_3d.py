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
from geometry_msgs.msg import PoseArray, PoseStamped, Point, PointStamped
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
import message_filters

class PERSON_detection():
	def __init__(self):
		#self.node_name = rospy.get_name()
		#rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.bridge = CvBridge()
		self.is_compressed = False
		self.frame_id = 'camera_link'
		# Image definition
		self.width = 640
		self.height = 480
		self.labels = ['background', 'person', 'palm']
		self.prob_threshold = 0.8
		self.objects = []
		self.net = build_ssd('test', 300, len(self.labels))    # initialize SSD
		self.net.load_weights('/home/user/argbot/catkin_ws/src/track/src/argbot_58000.pth')
		if torch.cuda.is_available():
			self.net = self.net.cuda()
		msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=None)
		self.fx = msg.P[0]
		self.fy = msg.P[5]
		self.cx = msg.P[2]
		self.cy = msg.P[6]

		self.pre_time = rospy.get_time()
		self.reset_time = rospy.get_time()

		#img = cv2.imread("radio.jpg")
		self.image_pub = rospy.Publisher("/predict_img", Image, queue_size = 1)
		self.pub_path = rospy.Publisher("/person_path", Marker, queue_size = 1)
		self.pub_point = rospy.Publisher("/person_point", Marker, queue_size = 1)
		self.pub_tracking = rospy.Publisher("/tracking_point", Marker, queue_size = 1)
		# self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_cb, queue_size=1, buff_size = 2**24)
		image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
		depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
		ts = message_filters.TimeSynchronizer([image_sub, depth_sub], 10)
		ts.registerCallback(self.callback)
		self.tracker = Tracker.Tracker()
		self.aim_coordinate = None
		self.path = []
		self.person_list = []
		self.cv_depthimage = None
		self.start_tracking = False
		#self.predict(img)

	def callback(self, img_msg, depth_msg):
		try:
			if self.is_compressed:
				np_arr = np.fromstring(img_msg.data, np.uint8)
				cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
				self.cv_depthimage = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
			else:
				cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
				self.cv_depthimage = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
			
			(rows, cols, channels) = cv_image.shape
			self.width = cols
			self.height = rows
			predict_img = self.predict(cv_image)
			if self.start_tracking:
				coordinate = self.tracking()
				if coordinate is not None:
					self.draw_tracking(coordinate)
			self.pre_time = rospy.get_time()

			try:
				self.image_pub.publish(self.bridge.cv2_to_imgmsg(predict_img, "bgr8"))
			except CvBridgeError as e:
				print(e)

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
		# print(1./(t2-t1))
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

		self.person_list = []
		palm_bbx_list = []
		person_bbx_list = []

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
			if self.labels[obj[4]] == 'person':
				palm_cnt = palm_cnt + 1
				# self.path.append(list(center))
				c_depth = self.center_depth(center)
				coordinate = self.getXYZ(center[0], center[1], c_depth)
				self.person_list.append(coordinate)
				person_bbx_list.append(bbx)
				self.draw_point(coordinate)
			if self.labels[obj[4]] == 'palm':
				palm_bbx_list.append(bbx)

		reset_lock = rospy.get_time() - self.reset_time > 3

		if reset_lock and len(palm_bbx_list) == 1 and len(self.person_list) > 0 and not self.start_tracking:
			print("Start tracking")
			iou_list = []
			for person_bbx in person_bbx_list:
				iou_list.append(self.get_iou(person_bbx, palm_bbx_list[0]))
			iou_list = np.array(iou_list)
			goal_idx = np.where(iou_list == iou_list.max())[0][0]
			self.aim_coordinate = self.person_list[goal_idx][:2]
			self.start_tracking = True
			self.tracker.Update(self.aim_coordinate)
		elif len(palm_bbx_list) == 2 and self.start_tracking:
			print("Reset tracking")
			self.reset_time = rospy.get_time()
			self.start_tracking = False

		# if palm_cnt > 1:
		# 	self.path = []
		# draw_path = np.array(self.path, np.int32)
		# draw_path = draw_path.reshape((-1,1,2))
		# cv2.polylines(img, [draw_path], False, (0,255,255))
		return img

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

	def get_iou(self, bbx1, bbx2): # bbx = [x1, y1, x2, y2]
		area1 = abs((bbx1[0] - bbx1[2])*(bbx1[1] - bbx1[3]))
		area2 = abs((bbx2[0] - bbx2[2])*(bbx2[1] - bbx2[3]))
		dx = min(bbx1[2], bbx2[2]) - max(bbx1[0], bbx2[0])
		dy = min(bbx1[3], bbx2[3]) - max(bbx1[1], bbx2[1])
		if (dx >= 0) and (dy >= 0):
			return dx*dy/min(area1, area2)

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

	def tracking(self):
		if len(self.person_list) == 0:
			return
		dt = rospy.get_time() - self.pre_time
		self.tracker.set_dt(dt)
		self.aim_coordinate = self.tracker.get_prediction()[0]
		dis_list = []
		for person in self.person_list:
			dis = self.distance(person[:2], self.aim_coordinate)
			dis_list.append(dis)
		dis_list = np.array(dis_list)
		aim_idx = np.where(dis_list == dis_list.max())[0][0]
		self.tracker.Update(self.person_list[aim_idx][:2])
		return self.aim_coordinate

	def distance(self, a, b):
		return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

	def draw_path(self):
		'''marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.type = marker.LINE_STRIP
		marker.action = marker.ADD
		marker.scale.x = 0.3
		marker.scale.y = 0.3
		marker.scale.z = 0.3
		marker.color.a = 1.0
		marker.color.r = 1.0
		marker.color.g = 1.0
		marker.color.b = 0
		marker.pose.orientation.x = 0.0
		marker.pose.orientation.y = 0.0
		marker.pose.orientation.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.pose.position.x = 0.0
		marker.pose.position.y = 0.0
		marker.pose.position.z = 0.0
		marker.points = []
		for i in range(self.waypoint_size):
			p = Point()
			p.x = self.waypoint_list.list[i].x
			p.y = self.waypoint_list.list[i].y
			p.z = self.waypoint_list.list[i].z
			marker.points.append(p)
		self.pub_path.publish(marker)'''

	def draw_point(self, coordinate):
		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.type = marker.POINTS
		marker.pose.orientation.w = 1
		marker.scale.x = 0.1
		marker.scale.y = 0.1
		marker.scale.z = 0.1
		marker.color.a = 0.6
		marker.color.g = 1.0
		p = Point()
		p.x = coordinate[0]
		p.y = coordinate[1]
		marker.points = [p]
		self.pub_point.publish(marker)

	def draw_tracking(self, coordinate):
		marker = Marker()
		marker.header.frame_id = self.frame_id
		marker.type = marker.POINTS
		marker.pose.orientation.w = 1
		marker.scale.x = 0.2
		marker.scale.y = 0.2
		marker.scale.z = 0.2
		marker.color.a = 1.0
		marker.color.r = 1.0
		p = Point()
		p.x = coordinate[0]
		p.y = coordinate[1]
		marker.points = [p]
		self.pub_tracking.publish(marker)

	def onShutdown(self):
		rospy.loginfo("Shutdown.")


if __name__ == '__main__':
	rospy.init_node('PERSON_detection')
	foo = PERSON_detection()
	rospy.spin()
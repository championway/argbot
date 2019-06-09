#!/usr/bin/env python

from InstanceSeg_net import *
from pointnet import *
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import cv2
import rospy
import ctypes
from bb_pointnet.msg import *
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String,Float64, Bool, Header
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from sensor_msgs import point_cloud2
import pcl
import time
import os
from bb_pointnet.srv import *

class bb_pointnet(object):
	def __init__(self):
		self.subt_CLASSES =  [  # always index 0
				'bb_extinguisher', 'bb_drill']

		self.fx = 618.2425537109375
		self.fy = 618.5384521484375
		self.cx = 327.95947265625
		self.cy = 247.670654296875

		self.cv_bridge = CvBridge() 
		self.num_points = 60000
		
		#self.network = InstanceSeg(num_points = self.num_points)
		self.network = PointNetDenseCls(k = 2) 
		self.network = self.network.cuda()
		model_dir = "/home/andyser/code/subt_related/subt_arti_searching/instance/weights"
		model_name = "pointnet_epoch_90.pkl"	
		state_dict = torch.load(os.path.join(model_dir, model_name))
		self.network.load_state_dict(state_dict)
		self.origin = rospy.Publisher('/origin', PointCloud2, queue_size=10)
		self.prediction = rospy.Publisher('/prediction', PointCloud2, queue_size=10)
		self.predict_ser = rospy.Service("/pointnet_whole_scene", pointnet_whole_scene, self.callback)

	def callback(self, req):
		self.network.eval()
		pc_msg = rospy.wait_for_message("/pc_preprocessing",PointCloud2)
		points_list = []
		color_list = []

		for data in point_cloud2.read_points(pc_msg, skip_nans=True):
			points_list.append([data[0], data[1], data[2]])	
			color_list.append(data[3])

		point = np.asarray(points_list)
		color_list = np.asarray(color_list)
		if point.shape[0] < self.num_points:
			row_idx = np.random.choice(point.shape[0], self.num_points, replace=True)
		else:
			row_idx = np.random.choice(point.shape[0], self.num_points, replace=False)	

		point_in = torch.from_numpy(point[row_idx])  	## need to revise
		color_list = color_list[row_idx]

		point_in = np.transpose(point_in, (1, 0))
		point_in = point_in[np.newaxis,:]

		inputs = Variable(point_in).type(torch.FloatTensor).cuda()
		output = self.network(inputs)[0][0]

		_point_list = []
		_origin_list = []
		for i in range(self.num_points):
			s = struct.pack('>f' ,color_list[i])
			k = struct.unpack('>l',s)[0]
			pack = ctypes.c_uint32(k).value
			r = (pack & 0x00FF0000)>> 16
			g = (pack & 0x0000FF00)>> 8
			b = (pack & 0x000000FF)
			rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0] 
			#if output[i].argmax() == labels[i] and output[i].argmax() == 1:
			if output[i].argmax() == 1:
			#if output[i][0] < -0.1:
				_point_list.append([inputs[0][0][i], inputs[0][1][i], inputs[0][2][i],rgb])
			_origin_list.append([inputs[0][0][i], inputs[0][1][i], inputs[0][2][i],rgb])

		header = Header()
		header.stamp = rospy.Time.now()
		header.frame_id ="camera_link"

		fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1), PointField('z', 8, PointField.FLOAT32, 1), PointField('rgb', 12, PointField.UINT32, 1)]
		pointcloud_pre = point_cloud2.create_cloud(header, fields, _point_list)
		pointcloud_origin = point_cloud2.create_cloud(header, fields, _origin_list)

		self.prediction.publish(pointcloud_pre)
		self.origin.publish(pointcloud_origin)

		return "Finish" + ", point num: " + str(len(_point_list)) + ", origin num: " + str(point.shape[0])

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
	rospy.init_node('bb_pointnet_prediction',anonymous=False)
	bb_pointnet = bb_pointnet()
	rospy.on_shutdown(bb_pointnet.onShutdown)

	# while(1):
	# 	bb_pointnet.callback()

	rospy.spin()
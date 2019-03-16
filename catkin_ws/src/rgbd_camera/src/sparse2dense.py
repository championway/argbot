#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, PoseStamped, Point
import rospkg
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import message_filters

import os
import math
import time
import sys
import PIL
from models import *
import torchvision.transforms as transforms
from torch.autograd import Variable
#from models import *
import torch.nn as nn
import torch.nn.functional as F
import torch

class SPARSE2DENSE():
	def __init__(self):
		#rospy.loginfo("[%s] Initializing " %(self.node_name))
		self.bridge = CvBridge()
		self.cuda = True if torch.cuda.is_available() else False
		self.generator = GeneratorUNet(in_channels=1, out_channels=1)
		if self.cuda:
			self.generator = self.generator.cuda()
		self.generator.load_state_dict(torch.load('/home/arg_ws3/PyTorch-GAN/implementations/pix2pix/saved_models/sparse2dense/generator_190.pth'))
		self.cv_depthimage = None
		self.generate_img = None
		self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
		self.data_transform = transforms.Compose([transforms.Resize((256, 256), 
												  PIL.Image.BICUBIC),
												  transforms.ToTensor()])
		#-------point cloud without color-------
		self.depth_sub = rospy.Subscriber("/dp_img", Image, self.img_cb, queue_size = 1, buff_size = 2**24)
		#self.depth_sub = rospy.Subscriber("/X1/rgbd_camera/depth/image_raw", Image, self.img_cb, queue_size = 1, buff_size = 2**24)
		#------------------------------------

		#-------point cloud with color-------
		#self.depth_sub = message_filters.Subscriber("/X1/rgbd_camera/depth/image_raw", Image)
		#self.image_sub = message_filters.Subscriber("/X1/rgbd_camera/rgb/image_raw", Image)
		#self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], 1, 0.3)
		#self.ts.registerCallback(self.img_cb)
		#------------------------------------

		#self.pc_pub = rospy.Publisher("/pointcloud2_transformed", PointCloud2, queue_size=1)
		self.image_pub = rospy.Publisher("/generate_dp", Image, queue_size = 1)
		self.points = []
		rospy.loginfo("Start Generating depth image")

	def img_cb(self, depth_data):
		self.cv_depthimage = self.bridge.imgmsg_to_cv2(depth_data, "32FC1")
		self.generate_image()
		self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.generate_img, "32FC1"))

	def generate_image(self):
		prev_time = time.time()
		pil_im = PIL.Image.fromarray(self.cv_depthimage)
		pil_im = self.data_transform(pil_im)
		pil_im = pil_im.unsqueeze(0)

		my_img = Variable(pil_im.type(self.Tensor))
		my_img_fake = self.generator(my_img)
		my_img_fake = my_img_fake.squeeze(0).detach().cpu()
		pil_ = my_img_fake.mul(255).clamp(0, 255).to(torch.float32).permute(1, 2, 0)
		#pil_ = my_img_fake.mul(255).clamp(0, 255).byte().permute(1, 2, 0)
		pil_ = np.array(pil_)
		pil_ = pil_[...,::-1]

		#pil_ = np.float32(pil_)
		self.generate_img = cv2.resize(pil_, (640, 480))
		#print("Hz: ", 1./(time.time() - prev_time))

if __name__ == '__main__':
	rospy.init_node('SPARSE2DENSE')
	foo = SPARSE2DENSE()
	rospy.spin()
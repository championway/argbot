#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
from sensor_msgs.msg import Image, LaserScan
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseArray, Point
from visualization_msgs.msg import Marker, MarkerArray
import rospkg
from cv_bridge import CvBridge, CvBridgeError
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
from PIL import Image

class AlexNet(nn.Module):
	def __init__(self, base_model, num_classes):
		super(AlexNet, self).__init__()
		# Everything except the last linear layer
		self.base_features = base_model.features
		self.base_classifier = nn.Sequential(*base_model.classifier[:-1])
		self.new_classifier = nn.Sequential(
			nn.Linear(4096, num_classes)
		)
		self.modelName = 'AlexNet'

		'''
		# Freeze those weights
		for p in self.features.parameters():
			p.requires_grad = False
		'''

	def forward(self, img):
		f = self.base_features(img)
		f = f.view(f.size(0), -1)
		fc2 = self.base_classifier(f)
		y = self.new_classifier(fc2)
		return y



class classify():
	def __init__(self):
		self.node_name = rospy.get_name()
		rospy.loginfo("[%s] Initializing " %(self.node_name))
		rospy.Subscriber('/X1/scan', LaserScan, self.call_back, queue_size = 1, buff_size = 2**24)
		self.bin = 360
		self.range_max = 5.5
		self.border = 10
		self.scale = 7.
		self.point_size = 2
		self.t_sum = 0.
		self.env_change_param = 20
		self.env_count = 0
		self.past_predict = None
		self.past_state = ""
		self.env_state = [] 
		#self.index = 0

		# ***************************************************************
		# Get the position of caffemodel folder
		# ***************************************************************
		#self.model_name = rospy.get_param('~model_name')
		model_name = "epoch45"
		rospy.loginfo('[%s] model name = %s' %(self.node_name, model_name))
		#model_path = rospack.get_path('classification') + '/model/' + model_name + '.pth'
		model_path = '/home/arg_ws3/argbot/catkin_ws/src/classify/models/' + model_name + '.pth'
		self.labels = ['cliff', 'end', 'four', 'straight']

		# ***************************************************************
		# Set up deeplearning model
		# ***************************************************************
		self.data_transform = transforms.Compose([ \
			transforms.Resize(227), \
			#transforms.RandomHorizontalFlip(), \
			transforms.ToTensor(), \
			transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                 std=[0.229, 0.224, 0.225])])
		self.model = torchvision.models.alexnet(pretrained = False)
		self.model.classifier[6] = nn.Linear(4096, 4)
		self.model = self.model.cuda()
		base_model = torchvision.models.alexnet(pretrained = False)
		self.model = AlexNet(base_model, 4)
		self.model = self.model.cuda()
		self.model.load_state_dict(torch.load(model_path))
		print(self.model)
		print("--------------------")
		print("Start to predict environment")
		print("--------------------\n\n")
		#self.model =  torch.load(model_path)
		
	def init_param(self):
		self.width = int(self.range_max*self.scale*2 + self.border*2)
		self.height = int(self.range_max*self.scale*2 + self.border*2)
		self.img = np.zeros((int(self.height), int(self.width)), np.uint8)

	def call_back(self, msg):
		self.range_max = msg.range_max
		self.bin = len(msg.ranges)
		self.init_param()

		for i in range(self.bin):
			if msg.ranges[i] != float("inf"):
				rad = (i/360.)*2.*math.pi/2.
				x = self.scale*msg.ranges[i] * np.cos(rad)
				y = self.scale*msg.ranges[i] * np.sin(rad)
				x_ = int(x + self.width/2.)
				y_ = int(y + self.height/2.)
				self.img[x_][y_] = 255
				self.img = self.img_dilate(self.img, x_, y_)
		environment = self.classify()
		self.check_environment(environment)

	def check_environment(self, environment):
		if environment == self.past_predict:
			self.env_count = self.env_count + 1
		else:
			self.env_count = 0

		if self.env_count > self.env_change_param:
			if environment != self.past_state:
				self.past_state = environment
				self.env_state.append(environment)
				sys.stdout.write('\rSTATE: ')
				for state in self.env_state:
					sys.stdout.write(state + " --> ")
				sys.stdout.flush()
		self.past_predict = environment

	def img_dilate(self, img, x, y):
		for m in range(-self.point_size, self.point_size + 1):
			for n in range(-self.point_size, self.point_size + 1):
				img[x + m][y + n] = 255
		return img

	def classify(self):
		# ***************************************************************
		# Using Pytorch Model to do prediction
		# ***************************************************************
		#scale_tensor = [torch.FloatTensor([[i]]).cuda() for i in self.scale]
		#cv_img = cv2.resize(self.image, self.dim)
		pil_img = Image.fromarray(self.img.astype('uint8'))
		torch_img = self.data_transform(pil_img)
		torch_img = np.expand_dims(torch_img, axis=0)
		input_data = torch.tensor(torch_img).type('torch.FloatTensor').cuda()
		t_start = time.clock()
		input_data = input_data.repeat(1, 3, 1, 1) # from 1-channel image to 3-channel image
		output = self.model(input_data)
		pred_y = int(torch.max(output, 1)[1].cpu().data.numpy())
		t_duration = float(time.clock() - t_start)
		#self.count += 1
		#print "prediction time taken = ", t_duration
		#print "Predict: ", self.labels[pred_y]
		#print output_prob[output_max_class]
		#if output_prob[output_max_class]<0.7:
		#	return "None"
		return self.labels[pred_y]

if __name__ == '__main__':
	rospy.init_node('classify')
	foo = classify()
	rospy.spin()
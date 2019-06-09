#!/usr/bin/env python
import numpy as np
import cv2
import struct
import math
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from ssd import build_ssd

cap = cv2.VideoCapture(0)
labels = ['background', 'people', 'palm']
prob_threshold = 0.99
net = build_ssd('test', 300, len(labels))    # initialize SSD
net.load_weights('/home/arg_ws3/ssd.pytorch/weights/argbot/argbot_54000.pth')
if torch.cuda.is_available():
	net = net.cuda()

def main():
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()

		# Our operations on the frame come here
		img = predict(frame)
		# Display the resulting frame
		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


def predict(img):
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
	y = net(xx)
	scale = torch.Tensor(img.shape[1::-1]).repeat(2)
	detections = y.data	# torch.Size([1, 4, 200, 5]) --> [batch?, class, object, coordinates]
	objs = []
	for i in range(detections.size(1)): # detections.size(1) --> class size
		for j in range(5):	# each class choose top 5 predictions
			if detections[0, i, j, 0].numpy() > prob_threshold:
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
		cv2.putText(img, labels[obj[4]], (int(obj[0] + obj[2]), int(obj[1])), 0, 1, color,2)
		#print(self.labels[obj[4]])
	#cv2.imshow('image',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return img

main()
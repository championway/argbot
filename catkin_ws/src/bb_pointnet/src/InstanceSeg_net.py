
#from datasets import DatasetImgNetAugmentation, DatasetImgNetEval # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import numpy as np

class InstanceSeg(nn.Module):
    def __init__(self, num_points=1024):
        super(InstanceSeg, self).__init__()

        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv4_2 = nn.Conv1d(128, 128, 1)
        self.conv4_3 = nn.Conv1d(128, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.conv6 = nn.Conv1d(1088, 512, 1)
        self.conv6_2 = nn.Conv1d(512, 512, 1)
        self.conv6_3 = nn.Conv1d(512, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv7_2 = nn.Conv1d(256, 256, 1)
        self.conv7_3 = nn.Conv1d(256, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, 128, 1)
        self.conv10 = nn.Conv1d(128, 2, 1)
        self.max_pool = nn.MaxPool1d(num_points)

    def forward(self, x):
        batch_size = x.size()[0] # (x has shape (batch_size, 4, num_points))

        out = F.relu(self.conv1(x)) # (shape: (batch_size, 64, num_points))
        out = F.relu(self.conv2(out)) # (shape: (batch_size, 64, num_points))
        point_features = out

        out = F.relu(self.conv3(out)) # (shape: (batch_size, 64, num_points))
        out = F.relu(self.conv4(out)) # (shape: (batch_size, 128, num_points))
        out = F.relu(self.conv4_2(out)) # (shape: (batch_size, 128, num_points))
        out = F.relu(self.conv4_3(out)) # (shape: (batch_size, 128, num_points))
        out = F.relu(self.conv5(out)) # (shape: (batch_size, 1024, num_points))
        global_feature = self.max_pool(out) # (shape: (batch_size, 1024, 1))

        global_feature_repeated = global_feature.repeat(1, 1, self.num_points) # (shape: (batch_size, 1024, num_points))
        out = torch.cat([global_feature_repeated, point_features], 1) # (shape: (batch_size, 1024+64=1088, num_points))

        out = F.relu(self.conv6(out)) # (shape: (batch_size, 512, num_points))
        out = F.relu(self.conv6_2(out)) # (shape: (batch_size, 512, num_points))
        out = F.relu(self.conv6_3(out)) # (shape: (batch_size, 512, num_points))
        out = F.relu(self.conv7(out)) # (shape: (batch_size, 256, num_points))
        out = F.relu(self.conv7_2(out)) # (shape: (batch_size, 256, num_points))
        out = F.relu(self.conv7_3(out)) # (shape: (batch_size, 256, num_points))
        out = F.relu(self.conv8(out)) # (shape: (batch_size, 128, num_points))
        out = F.relu(self.conv9(out)) # (shape: (batch_size, 128, num_points))

        out = self.conv10(out) # (shape: (batch_size, 2, num_points))

        out = out.transpose(2,1).contiguous() # (shape: (batch_size, num_points, 2))
        out = F.log_softmax(out.view(-1, 2), dim=1) # (shape: (batch_size*num_points, 2))
        out = out.view(batch_size, self.num_points, 2) # (shape: (batch_size, num_points, 2))

        return out


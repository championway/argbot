#!/usr/bin/env python
import cv2
import numpy as np
import roslib
import rospy
import torch
import math

from pose_lib.with_mobilenet import PoseEstimationWithMobileNet
from pose_lib.keypoints import extract_keypoints, group_keypoints
from pose_lib.load_state import load_state
from pose_lib.pose import propagate_ids
from pose_lib.util import normalize, pad_width
import pose_lib.pose # for pose

from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge, CvBridgeError
from pose_msgs.msg import HumanPoses

class PoseDetector():
    def __init__(self):
        self.node_name = rospy.get_name()
        self.net_init()
        self.time = None
        self.bridge = CvBridge()
        self.is_compressed = False
        self.image_pub = rospy.Publisher("/pose_img", Image, queue_size = 1)
        self.pose_pub = rospy.Publisher("/human_pose", HumanPoses, queue_size = 1)
        self.image_sub = rospy.Subscriber("/camera/color/image_rect_color", Image, self.img_cb, queue_size=1, buff_size = 2**24)

    def net_init(self):
        self.cpu = False
        self.track = False
        self.stride = 8
        self.upsample_ratio = 4
        self.height_size = 256
        self.num_keypoints = pose_lib.pose.Pose.num_kpts
        self.net = PoseEstimationWithMobileNet()
        checkpoint = torch.load("/home/david/Downloads/checkpoint_iter_370000.pth", map_location='cpu')
        load_state(self.net, checkpoint)
        self.net = self.net.eval()
        if not self.cpu:
            self.net = self.net.cuda()

    def img_cb(self, msg):
        try:
            if self.is_compressed:
                np_arr = np.fromstring(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.time = msg.header.stamp
        (rows, cols, channels) = cv_image.shape
        self.width = cols
        self.height = rows
        self.detector(cv_image)

    def detector(self, img):
        previous_poses = []
        #for img in image_provider:
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(self.net, img, self.height_size, self.stride, self.upsample_ratio, self.cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(self.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[0]) / scale
        current_poses = []
        human_poses = HumanPoses()
        human_poses.size = len(pose_entries)
        # len(pose_entries) == number of people
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((self.num_keypoints, 2), dtype=np.int32) * -1
            human_pose = PoseArray()
            for kpt_id in range(self.num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                p = Pose()
                p.position.x = pose_keypoints[kpt_id, 0]
                p.position.y = pose_keypoints[kpt_id, 1]
                human_pose.poses.append(p)
            human_poses.pose_list.append(human_pose)
            
            pose = pose_lib.pose.Pose(pose_keypoints, pose_entries[n][18])
            #print('kpts: ', pose_keypoints)
            current_poses.append(pose)
            pose.draw(img)
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        if human_poses.size != 0:
            human_poses.header.stamp = self.time
            self.pose_pub.publish(human_poses)
        '''
        if self.track == True:
            propagate_ids(previous_poses, current_poses)
            previous_poses = current_poses
            for pose in current_poses:
                cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        '''
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e:
            print(e)



def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=float(1./256.)):
    height, width, _ = img.shape
    scale = float(net_input_height_size) / float(height)

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad


if __name__ == '__main__':
    rospy.init_node('PoseDetector')
    foo = PoseDetector()
    rospy.spin()
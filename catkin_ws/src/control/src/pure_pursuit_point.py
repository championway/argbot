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
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point, Twist
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry, Path
import rospkg
from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.server import Server
from control.cfg import pos_PIDConfig, ang_PIDConfig
from std_srvs.srv import SetBool, SetBoolResponse
robot = None
path = []
pub_point = rospy.Publisher('/pursue_point', PoseStamped, queue_size=10)

def odom_cb(msg):
	global robot, path, pub_point
	pose = PoseStamped()
	robot = [msg.pose.pose.position.x, msg.pose.pose.position.y]
	path_hold = []
	if path == []:
		return
	for p in path:
		start = False
		if start or distanceBtwnPoints(p[0], p[1], robot[0], robot[1]) > 2:
			start = True
			path_hold.append(p)
	path[:] = path_hold[:]
	if path == []:
		return
	pose.header = msg.header
	pose.pose.position.x = path[0][0]
	pose.pose.position.y = path[0][1]
	pub_point.publish(pose)

def path_cb(msg):
	global robot, path
	if robot is None:
		return
	path = []
	start = False
	for pose in msg.poses:
		p = [pose.pose.position.x, pose.pose.position.y]
		too_close = distanceBtwnPoints(p[0], p[1], robot[0], robot[1]) < 2
		if start or too_close:
			start = True
			path.append(p)

def distanceBtwnPoints(x1, y1, x2, y2):
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def main():
	#global robot, pub_point, path
	#path = []
	#robot = None
	rospy.init_node('pub_point', anonymous=True)
	rospy.Subscriber("/planning_path", Path, path_cb, queue_size=1)
	rospy.Subscriber('/odometry/ground_truth', Odometry, odom_cb, queue_size = 1, buff_size = 2**24)
	#pub_point = rospy.Publisher('/pursue_point', PoseStamped, queue_size=10)
	rospy.spin()

if __name__ == '__main__':
	main()
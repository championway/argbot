#!/usr/bin/python

import roslib   #roslib.load_manifest(PKG)
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
import glob, os
from tf.transformations import euler_from_quaternion

class ImageCreator():


    def __init__(self):
    	self.start = 0
        self.train_or_test = 0
        self.write_image = 0
        self.image_num = 0
        self.train_arr = np.zeros(15)
        self.test_arr = np.zeros(15)
        self.omega = 0
        self.omega_gain = 8.5 #have to modified everytime, since the omega gain isn't the same for everycar.
        self.bridge = CvBridge()
        self.hostname = ""
        self.np_arr = 0
        self.folder_name = '/argbot/catkin_ws/src/tracking/src' #write your folder name betweer ' /'
        is_compressed = False
        #---bag part---
        #os.chdir("Downloads")
        data_file = open("data_M.txt", "w")
        for file in glob.glob("/home/arg_ws3/argbot/catkin_ws/src/tracking/src/2019-02-18-14-45-17.bag"):
            with rosbag.Bag(file, 'r') as bag: #open first .bag
                print (file)
                for topic, msg, t in bag.read_messages():
                    left, right = '_', '_'
                    x, y, euler = '_', '_', '_'
                    if self.start == 0:
                        self.get_host_name(topic)
                        self.start = 1
                        if self.hostname == "":
                            print "unable to find hostname, finish execution"
                            exit()
                        else:
                        	print "start transforming bag to txt."

                    if topic == "/MONICA/cmd_drive":
                        left = msg.left
                        right = msg.right
                    if topic == "/MONICA/localization_gps_imu/odometry":
                        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                        euler = euler_from_quaternion(q)[2]
                        x = msg.pose.pose.position.x
                        y = msg.pose.pose.position.y
                    data_file.write(str('%d.%d'%(t.secs, t.nsecs))+",")
                    data_file.write(str(left)+","+str(right)+",")
                    data_file.write(str(x)+","+str(y)+","+str(euler))
                    data_file.write("\n")
        data_file.close()
    def get_host_name(self,topic): 
        arr = topic.split('/')
        self.hostname = arr[1]
        print "hostname get, it's: "+self.hostname

if __name__ == '__main__':

    #rospy.init_node(PKG)

    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass

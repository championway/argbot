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
        self.folder_name = '/media/arg_ws3/5E703E3A703E18EB/data/real_box/dark/' #write your folder name betweer ' /'
        is_compressed = False
        #---bag part---
        #os.chdir("Downloads")
        for file in glob.glob("/media/arg_ws3/5E703E3A703E18EB/data/real_box/kaohsiung_king_2_2019-02-14-19-06-31.bag"):
            with rosbag.Bag(file, 'r') as bag: #open first .bag
                print (file)
                for topic,msg,t in bag.read_messages():
                    if self.start == 0:
                        self.get_host_name(topic)
                        self.start = 1
                        if self.hostname == "":
                            print "unable to find hostname, finish execution"
                            exit()
                        else:
                        	print "start transforming bag to txt."
                    if topic == "/camera/color/image_rect_color":
                        print('s')
                        try:
                            if is_compressed:
                                self.np_arr = np.fromstring(msg.data, np.uint8)
                                cv_image = cv2.imdecode(self.np_arr, cv2.IMREAD_COLOR)
                            else:
                                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                        except CvBridgeError as e:
                            print(e)
                        image_name = "dark_" + str(self.image_num) + ".jpg" 
                        cv2.imwrite(self.folder_name + image_name, cv_image)
                        self.image_num = self.image_num + 1

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

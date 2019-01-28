#!/usr/bin/env python
import rospy
import roslib
import tf
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
def call_back(msg):
	# Global Frame: "/map"
	map_id = msg.header.frame_id
	br = tf.TransformBroadcaster()
	# map <---> gazebo coordinate
	#br.sendTransform((0.0, 0.0, 0.0),tf.transformations.quaternion_from_euler(0, 0, -np.pi/2.0), rospy.Time.now() ,map_id , "/X1/base_footprint")
	# robot <---> map
	quat = (msg.pose.pose.orientation.x,\
			msg.pose.pose.orientation.y,\
			msg.pose.pose.orientation.z,\
			msg.pose.pose.orientation.w)
	r, p, y = tf.transformations.euler_from_quaternion(quat)
	quat = tf.transformations.quaternion_from_euler (-r, -p, -y)
	br.sendTransform((-msg.pose.pose.position.x, -msg.pose.pose.position.y, -msg.pose.pose.position.z), \
					 (quat[0], quat[1], quat[2], quat[3]), rospy.Time.now(), map_id, "/X1/base_footprint")

if __name__=="__main__":
	# Tell ROS that we're making a new node.
	rospy.init_node("tf_publisher",anonymous=False)
	rospy.Subscriber("/X1/x1_velocity_controller/odom", Odometry, call_back, queue_size=1)
	rospy.spin()
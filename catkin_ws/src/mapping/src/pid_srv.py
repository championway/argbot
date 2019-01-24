#!/usr/bin/env python
import rospy

from dynamic_reconfigure.server import Server
from mapping.cfg import PIDConfig

def callback(config, level):
	rospy.loginfo("{Kp}, {Ki}, {Kd}".format(**config))
	return config

if __name__ == "__main__":
	rospy.init_node("PID_cfg", anonymous = True)
	srv = Server(PIDConfig, callback)
	rospy.spin()
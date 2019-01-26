#!/usr/bin/env python
import rospy
from tf import TransformListener,TransformerROS
from tf import LookupException, ConnectivityException, ExtrapolationException
import roslib
from sensor_msgs.msg import PointCloud2
from robotx_msgs.msg import ObjectPoseList
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import numpy as np

def call_back(msg):
    try:
        #print ("Process Object List")
        obj_list = ObjectPoseList()
        obj_list = msg
        position, quaternion = tf_.lookupTransform( "/odom", "/velodyne",rospy.Time(0))
        transpose_matrix = transformer.fromTranslationRotation(position, quaternion)
        robot_pose = np.dot(transpose_matrix, [0, 0, 0, 1])
        obj_list.robot.position.x = robot_pose[0]
        obj_list.robot.position.y = robot_pose[1]
        obj_list.robot.position.z = robot_pose[2]
        obj_list.robot.orientation.x = quaternion[0]
        obj_list.robot.orientation.y = quaternion[1]
        obj_list.robot.orientation.z = quaternion[2]
        obj_list.robot.orientation.w = quaternion[3]
        for obj_index in range(obj_list.size):
            center_x = obj_list.list[obj_index].position.x
            center_y = obj_list.list[obj_index].position.y
            center_z = obj_list.list[obj_index].position.z
            center  = np.array([center_x, center_y, center_z, 1])
            new_center = np.dot(transpose_matrix, center)
            obj_list.list[obj_index].position.x = new_center[0]
            obj_list.list[obj_index].position.y = new_center[1]
            obj_list.list[obj_index].position.z = new_center[2]
        obj_list.header.frame_id = "/odom"
        pub_obj.publish(obj_list)

    except (LookupException, ConnectivityException, ExtrapolationException):
        print "Nothing Happen"

if __name__ == "__main__":
    # Tell ROS that we're making a new node.
    rospy.init_node("object_map",anonymous=False)
    tf_ = TransformListener()
    transformer = TransformerROS()
    sub_classify = rospy.get_param('~classify', False)
    if sub_classify:
        sub_topic = "/obj_list/classify"
    else:
        sub_topic = "/obj_list"
    print "Object map subscribe from: ", sub_topic
    rospy.Subscriber(sub_topic, ObjectPoseList, call_back, queue_size=10)
    #rospy.Subscriber("/waypointList", WaypointList, call_back, queue_size=10)
    pub_obj = rospy.Publisher("/obj_list/odom", ObjectPoseList, queue_size=1)
    #pub_rviz = rospy.Publisher("/wp_path", Marker, queue_size = 1)
    rospy.spin()
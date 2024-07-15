#!/usr/bin/env python
import roslib, sys, rospy, os
from std_msgs.msg import String, Int32

def callback(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("Hello: I am encountering: ", data.data)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('exercise_performance_subscriber', anonymous=True)

    rospy.Subscriber("/set_performance", Int32, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
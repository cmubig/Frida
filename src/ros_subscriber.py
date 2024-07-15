#!/usr/bin/env python
import roslib, sys, rospy, os
from std_msgs.msg import String, Int32

def callback(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("Hello: I am encountering: ", data.data)
    
def listener():

    rospy.init_node('exercise_performance_subscriber', anonymous=True)

    rospy.Subscriber("/set_performance", Int32, callback)

    print("hello, I am at part 1")
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    print("hello, I am at part 2")

def listen_once():

    rospy.init_node('exercise_performance_subscriber', anonymous=True)
    
    print("hello I reached part 1")
    msg = rospy.wait_for_message("/set_performance", Int32, timeout=None)

    print("hello I reached part 2")
    print("This was the message: ", msg)
    print("hello I reached part 3")
    

if __name__ == '__main__':
    # listener()
    listen_once()
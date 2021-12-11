#! /usr/bin/env python



# def global_to_canvas_coordinates(x,y,z):
#     x_new = x + CANVAS_POSITION[0]/2
#     y_new = y - CANVAS_POSITION[1]
#     z_new = z
#     return x_new, y_new, z_new

def canvas_to_global_coordinates(x,y,z):
    x_new = (x -.5) * CANVAS_WIDTH + CANVAS_POSITION[0]
    y_new = y*CANVAS_HEIGHT + CANVAS_POSITION[1]
    z_new = z
    return x_new, y_new, z_new

# import argparse
# import importlib
# import os
# import numpy as np

# import rospy
# from geometry_msgs.msg import (
#     PoseStamped,
#     Pose,
#     Point,
#     Quaternion,
# )
# from std_msgs.msg import Header
# from sensor_msgs.msg import JointState

# from intera_core_msgs.srv import (
#     SolvePositionIK,
#     SolvePositionIKRequest,
#     SolvePositionFK,
#     SolvePositionFKRequest,
# )
# import intera_interface
# from intera_interface import CHECK_VERSION

# def good_morning_robot():
#     print("Getting robot state... ")
#     rs = intera_interface.RobotEnable(False)
#     init_state = rs.state().enabled
#     print("Enabling robot... ")
#     rs.enable()

#     def clean_shutdown():
#         """
#         Exits example cleanly by moving head to neutral position and
#         maintaining start state
#         """
#         print("\nExiting example...")
#         limb = intera_interface.Limb(synchronous_pub=True)
#         limb.move_to_neutral(speed=.15)

#     rospy.on_shutdown(clean_shutdown)
#     print("Excecuting... ")

#     return rs

# def good_night_robot():
#     """ Tuck it in, read it a story """
#     rospy.signal_shutdown("Example finished.")
#     print("Done")

# def get_joint_angles():
#     limb = intera_interface.Limb()
#     return limb.joint_angles()
#     # return limb.joint_ordered_angles()

# def forward_kinematics(joint_angles):
#     limb = intera_interface.Limb()
#     return limb.fk_request(joint_angles)
#     #return limb.endpoint_pose()
#     # return limb.joint_angles_to_cartesian_pose(joint_angles)


# def fk_service_client(joint_angles, limb = "right"):
#     ns = "ExternalTools/" + limb + "/PositionKinematicsNode/FKService"
#     fksvc = rospy.ServiceProxy(ns, SolvePositionFK)
#     fkreq = SolvePositionFKRequest()
#     joints = JointState()
#     joints.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
#                    'right_j4', 'right_j5', 'right_j6']
#     joints.position = [0.763331, 0.415979, -1.728629, 1.482985,
#                        -1.135621, -1.674347, -0.496337]

#     joints.name = list(joint_angles.keys())
#     joints.position = list(joint_angles.values())

#     pos = {}
#     for i in range(len(joints.name)):
#         name = joints.name[i]
#         position = joints.position[i]
#         # print(name, position)
#         pos[name] = position
#     print('asdfasdf', pos)

#     # Add desired pose for forward kinematics
#     fkreq.configuration.append(joints)
#     # Request forward kinematics from base to "right_hand" link
#     fkreq.tip_names.append('right_hand')

#     try:
#         rospy.wait_for_service(ns, 20.0)
#         resp = fksvc(fkreq)
#     except (rospy.ServiceException, rospy.ROSException) as e:
#         rospy.logerr("Service call failed: %s" % (e,))
#         return False

#     # Check if result valid
#     if (resp.isValid[0]):
#         rospy.loginfo("SUCCESS - Valid Cartesian Solution Found")
#         rospy.loginfo("\nFK Cartesian Solution:\n")
#         rospy.loginfo("------------------")
#         rospy.loginfo("Response Message:\n%s", resp)
#     else:
#         rospy.logerr("INVALID JOINTS - No Cartesian Solution Found.")
#         rospy.loginfo("Response Message:\n%s", resp)
#         return False

#     # return True
#     # Result to dictionary of joint angles
#     pos = {}
#     for i in range(len(resp.joints[0].name)):
#         name = resp.joints[0].name[i]
#         position = resp.joints[0].position[i]
#         # print(name, position)
#         pos[name] = position
#     return pos


# def inverse_kinematics(position, orientation, seed_position=None, debug=False):
#     """
#     args:
#         position=(x,y,z)
#         orientation=(x,y,z,w)
#     kwargs:
#         seed_position={'right_j0':float, 'right_j1':float, ...}
#     return:
#         dict{'right_j0',float} - dictionary of joint to joint angle
#     """
#     limb = intera_interface.Limb()
#     return limb.ik_request(position, joint_seed=seed_position, nullspace_goal=None)

def move(limb, position, timeout=3, speed=0.1):
    """
    args:
        dict{'right_j0',float} - dictionary of joint to joint angle
    """
    # rate = rospy.Rate(100)
    try:
        # limb = intera_interface.Limb(synchronous_pub=False)
        # limb.move_to_neutral()

        # print('Positions:', position)
        limb.set_joint_position_speed(speed=speed)
        limb.move_to_joint_positions(position, timeout=timeout,
                                     threshold=0.008726646)
        limb.set_joint_position_speed(speed=.1)
        # rate.sleep()
    except Exception as e:
        print('Exception while moving robot:\n', e)



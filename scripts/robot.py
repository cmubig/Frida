#! /usr/bin/env python
import os

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

class Robot:
    '''
        Low-level action functionality of the robot.
        This is an abstract class, see its children for usage.
    '''
    def __init__(self, debug, node_name="painting"):
        self.debug_bool = debug
        import rospy
        rospy.init_node(node_name)

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        raise Exception("This method must be implemented")

    def good_night_robot(self):
        raise Exception("This method must be implemented")

    def move_to_joint_positions(self, position):
        raise Exception("This method must be implemented")

class SimulatedRobot(Robot, object):
    def __init__(self, debug=True):
        pass

    def good_morning_robot(self):
        pass

    def good_night_robot(self):
        pass

    def move_to_joint_positions(self, position):
        pass
    def inverse_kinematics(self, position, orientation, seed_position=None, debug=False):
        pass
    def move_to_joint_positions(self, position, timeout=3, speed=0.1):
        pass


class Sawyer(Robot, object):
    def __init__(self, debug=True):
        super(Sawyer, self).__init__(debug)
        import rospy


        from intera_core_msgs.srv import (
            SolvePositionIK,
            SolvePositionIKRequest,
            SolvePositionFK,
            SolvePositionFKRequest,
        )
        import intera_interface
        from intera_interface import CHECK_VERSION
        import PyKDL
        from tf_conversions import posemath

        self.limb = intera_interface.Limb(synchronous_pub=False)

        self.ns = "ExternalTools/right/PositionKinematicsNode/IKService"
        self.iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK, persistent=True)
        rospy.wait_for_service(self.ns, 5.0)

    def good_morning_robot(self):
        import intera_interface
        import rospy
        self.debug("Getting robot state... ")
        rs = intera_interface.RobotEnable(False)
        init_state = rs.state().enabled
        self.debug("Enabling robot... ")
        rs.enable()

        def clean_shutdown():
            """
            Exits example cleanly by moving head to neutral position and
            maintaining start state
            """
            self.debug("\nExiting example...")
            limb = intera_interface.Limb(synchronous_pub=True)
            limb.move_to_neutral(speed=.2)

        rospy.on_shutdown(clean_shutdown)
        self.debug("Excecuting... ")

        return rs

    def good_night_robot(self):
        import rospy
        """ Tuck it in, read it a story """
        rospy.signal_shutdown("Example finished.")
        self.debug("Done")

    # def inverse_kinematics(self, position, orientation, seed_position=None, debug=False):
    #     """
    #     args:
    #         position=(x,y,z)
    #         orientation=(roll,pitch,yaw)
    #     kwargs:
    #         seed_position={'right_j0':float, 'right_j1':float, ...}
    #     return:
    #         dict{'right_j0',float} - dictionary of joint to joint angle
    #     """
    #     ikreq = SolvePositionIKRequest()
    #     hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    #     # pose = PoseStamped(
    #     #     header=hdr,
    #     #     pose=Pose(
    #     #         position=Point(
    #     #             x=position[0],
    #     #             y=position[1],
    #     #             z=position[2],
    #     #         ),
    #     #         orientation=Quaternion(
    #     #             x=orientation[0],
    #     #             y=orientation[1],
    #     #             z=orientation[2],
    #     #             w=orientation[3],
    #     #         ),
    #     #     ),
    #     # )

    #     endpoint_state = self.limb.tip_state('right_hand')
    #     if endpoint_state is None:
    #         rospy.logerr('Endpoint state not found with tip name %s', args.tip_name)
    #         return None
    #     pose = endpoint_state.pose
    #     print(pose)

    #     # create kdl frame from relative pose
    #     #position= [.01, 0.02, 0.03]
    #     rot = PyKDL.Rotation.RPY(orientation[0],
    #                              orientation[1],
    #                              orientation[2])
    #     trans = PyKDL.Vector(position[0] - pose.position.x,
    #                          position[1] - pose.position.y,
    #                          position[2] - pose.position.z)
    #     f2 = PyKDL.Frame(rot, trans)
    #     # and convert the result back to a pose message
    #     # if args.in_tip_frame:
    #     #     # end effector frame
    #     #     pose = posemath.toMsg(posemath.fromMsg(pose) * f2)
    #     # else:
    #     # base frame
    #     pose = posemath.toMsg(f2 * posemath.fromMsg(pose))
    #     # pose.position = Point(
    #     #             x=position[0],
    #     #             y=position[1],
    #     #             z=position[2],
    #     #         )
    #     print(pose)
    #     poseStamped = PoseStamped()
    #     poseStamped.pose = pose
    #     pose = poseStamped

    #     # Add desired pose for inverse kinematics
    #     ikreq.pose_stamp.append(pose)
    #     # Request inverse kinematics from base to "right_hand" link
    #     ikreq.tip_names.append('right_hand')

    #     if (seed_position is not None):
    #         # Optional Advanced IK parameters
    #         # rospy.loginfo("Running Advanced IK Service Client example.")
    #         # The joint seed is where the IK position solver starts its optimization
    #         ikreq.seed_mode = ikreq.SEED_USER
    #         seed = JointState()
    #         seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
    #                      'right_j4', 'right_j5', 'right_j6']
    #         seed.position = [seed_position['right_j0'], seed_position['right_j1'],
    #                          seed_position['right_j2'], seed_position['right_j3'],
    #                          seed_position['right_j4'], seed_position['right_j5'],
    #                          seed_position['right_j6']]
    #         ikreq.seed_angles.append(seed)

    #         # # Once the primary IK task is solved, the solver will then try to bias the
    #         # # the joint angles toward the goal joint configuration. The null space is
    #         # # the extra degrees of freedom the joints can move without affecting the
    #         # # primary IK task.
    #         # ikreq.use_nullspace_goal.append(True)
    #         # # The nullspace goal can either be the full set or subset of joint angles
    #         # goal = JointState()
    #         # goal.name = ['right_j1', 'right_j2', 'right_j3']
    #         # goal.position = [0.1, -0.3, 0.5]
    #         # ikreq.nullspace_goal.append(goal)
    #         # # The gain used to bias toward the nullspace goal. Must be [0.0, 1.0]
    #         # # If empty, the default gain of 0.4 will be used
    #         # ikreq.nullspace_gain.append(0.4)

    #     try:
    #         resp = self.iksvc(ikreq)
    #     except (rospy.ServiceException, rospy.ROSException) as e:
    #         rospy.logerr("Service call failed: %s" % (e,))
    #         return False


    #     # if resp.result_type[0] == resp.IK_IN_COLLISION:
    #     #     print('COOLLISSSIIIIONNN')

    #     # Check if result valid, and type of seed ultimately used to get solution
    #     if (resp.result_type[0] > 0):
    #         seed_str = {
    #                     ikreq.SEED_USER: 'User Provided Seed',
    #                     ikreq.SEED_CURRENT: 'Current Joint Angles',
    #                     ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
    #                    }.get(resp.result_type[0], 'None')
    #         if debug:
    #             rospy.loginfo("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
    #                   (seed_str,))
    #         # Format solution into Limb API-compatible dictionary
    #         limb_joints = dict(list(zip(resp.joints[0].name, resp.joints[0].position)))
    #         if debug:
    #             rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
    #             rospy.loginfo("------------------")
    #             rospy.loginfo("Response Message:\n%s", resp)
    #     else:
    #         rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
    #         rospy.logerr("Result Error %d", resp.result_type[0])
    #         return False
    #     # Result to dictionary of joint angles
    #     pos = {}
    #     for i in range(len(resp.joints[0].name)):
    #         name = resp.joints[0].name[i]
    #         position = resp.joints[0].position[i]
    #         # print(name, position)
    #         pos[name] = position
    #     return pos

    def inverse_kinematics(self, position, orientation, seed_position=None, debug=False):
        """
        args:
            position=(x,y,z)
            orientation=(x,y,z,w)
        kwargs:
            seed_position={'right_j0':float, 'right_j1':float, ...}
        return:
            dict{'right_j0',float} - dictionary of joint to joint angle
        """
        import rospy
        from intera_core_msgs.srv import (
            SolvePositionIK,
            SolvePositionIKRequest,
            SolvePositionFK,
            SolvePositionFKRequest,
        )
        from geometry_msgs.msg import (
            PoseStamped,
            Pose,
            Point,
            Quaternion,
        )
        from std_msgs.msg import Header
        from sensor_msgs.msg import JointState
        ikreq = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')
        pose = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=position[0],
                    y=position[1],
                    z=position[2],
                ),
                orientation=Quaternion(
                    x=orientation[0],
                    y=orientation[1],
                    z=orientation[2],
                    w=orientation[3],
                ),
            ),
        )
        # Add desired pose for inverse kinematics
        ikreq.pose_stamp.append(pose)
        # Request inverse kinematics from base to "right_hand" link
        ikreq.tip_names.append('right_hand')

        if (seed_position is not None):
            # Optional Advanced IK parameters
            # rospy.loginfo("Running Advanced IK Service Client example.")
            # The joint seed is where the IK position solver starts its optimization
            try:
                ikreq.seed_mode = ikreq.SEED_USER
                seed = JointState()
                seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3',
                             'right_j4', 'right_j5', 'right_j6']
                seed.position = [seed_position['right_j0'], seed_position['right_j1'],
                                 seed_position['right_j2'], seed_position['right_j3'],
                                 seed_position['right_j4'], seed_position['right_j5'],
                                 seed_position['right_j6']]
                ikreq.seed_angles.append(seed)
            except Exception as e:
                print(e)

            # # Once the primary IK task is solved, the solver will then try to bias the
            # # the joint angles toward the goal joint configuration. The null space is
            # # the extra degrees of freedom the joints can move without affecting the
            # # primary IK task.
            # ikreq.use_nullspace_goal.append(True)
            # # The nullspace goal can either be the full set or subset of joint angles
            # goal = JointState()
            # goal.name = ['right_j1', 'right_j2', 'right_j3']
            # goal.position = [0.1, -0.3, 0.5]
            # ikreq.nullspace_goal.append(goal)
            # # The gain used to bias toward the nullspace goal. Must be [0.0, 1.0]
            # # If empty, the default gain of 0.4 will be used
            # ikreq.nullspace_gain.append(0.4)

        try:
            resp = self.iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False


        # if resp.result_type[0] == resp.IK_IN_COLLISION:
        #     print('COOLLISSSIIIIONNN')

        # Check if result valid, and type of seed ultimately used to get solution
        if (resp.result_type[0] > 0):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp.result_type[0], 'None')
            if debug:
                rospy.loginfo("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
                      (seed_str,))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(list(zip(resp.joints[0].name, resp.joints[0].position)))
            if debug:
                rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
                rospy.loginfo("------------------")
                rospy.loginfo("Response Message:\n%s", resp)
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            rospy.logerr("Result Error %d", resp.result_type[0])
            return False
        # Result to dictionary of joint angles
        pos = {}
        for i in range(len(resp.joints[0].name)):
            name = resp.joints[0].name[i]
            position = resp.joints[0].position[i]
            # print(name, position)
            pos[name] = position
        return pos

    def move_to_joint_positions(self, position, timeout=3, speed=0.1):
        """
        args:
            dict{'right_j0',float} - dictionary of joint to joint angle
        """
        # rate = rospy.Rate(100)
        try:
            # print('Positions:', position)
            self.limb.set_joint_position_speed(speed=speed)
            self.limb.move_to_joint_positions(position, timeout=timeout,
                                         threshold=0.008726646)
            self.limb.set_joint_position_speed(speed=.1)
            # rate.sleep()
        except Exception as e:
            print('Exception while moving robot:\n', e)
            import traceback
            import sys
            print(traceback.format_exc())


    def display_image(self, file_path):
        import intera_interface
        head_display = intera_interface.HeadDisplay()
        # display_image params:
        # 1. file Path to image file to send. Multiple files are separated by a space, eg.: a.png b.png
        # 2. loop Display images in loop, add argument will display images in loop
        # 3. rate Image display frequency for multiple and looped images.
        head_display.display_image(file_path, False, 100)
    def display_frida(self):
        import rospkg
        rospack = rospkg.RosPack()
        # get the file path for rospy_tutorials
        ros_dir = rospack.get_path('paint')
        self.display_image(os.path.join(str(ros_dir), 'scripts', 'frida.jpg'))

    def take_picture(self):
        import cv2
        from cv_bridge import CvBridge, CvBridgeError
        import matplotlib.pyplot as plt
        def show_image_callback(img_data):
            """The callback function to show image by using CvBridge and cv
            """
            bridge = CvBridge()
            try:
                cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
            except CvBridgeError as err:
                rospy.logerr(err)
                return

            # edge_str = ''
            # cv_win_name = ' '.join(['heyyyy', edge_str])
            # cv2.namedWindow(cv_win_name, 0)
            # refresh the image on the screen
            # cv2.imshow(cv_win_name, cv_image)
            # cv2.waitKey(3)
            plt.imshow(cv_image[:,:,::-1])
            plt.show()
        rp = intera_interface.RobotParams()
        valid_cameras = rp.get_camera_names()
        print('valid_cameras', valid_cameras)

        camera = 'head_camera'
        # camera = 'right_hand_camera'
        cameras = intera_interface.Cameras()
        if not cameras.verify_camera_exists(camera):
            rospy.logerr("Could not detect the specified camera, exiting the example.")
            return
        rospy.loginfo("Opening camera '{0}'...".format(camera))
        cameras.start_streaming(camera)
        cameras.set_callback(camera, show_image_callback,
            rectify_image=False)
        raw_input('Attach the paint brush now. Press enter to continue:')
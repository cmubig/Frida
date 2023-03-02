#! /usr/bin/env python
import os
import numpy as np

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

    def go_to_cartesian_pose(self, positions, orientations, precise=False):
        raise Exception("This method must be implemented")


class Franka(Robot, object):
    '''
        Low-level action functionality of the Franka robot.
    '''
    def __init__(self, debug, node_name="painting"):
        import sys
        sys.path.append('~/Documents/frankapy/frankapy/')
        from frankapy import FrankaArm

        self.debug_bool = debug
        self.fa = FrankaArm()

        # reset franka to its home joints
        self.fa.reset_joints()

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def good_morning_robot(self):
        # reset franka to its home joints
        self.fa.reset_joints()

    def good_night_robot(self):
        # reset franka back to home
        self.fa.reset_joints()

    def create_rotation_transform(pos, quat):
        from autolab_core import RigidTransform
        rot = RigidTransform.rotation_from_quaternion(quat)
        rt = RigidTransform(rotation=rot, translation=pos,
                from_frame='franka_tool', to_frame='world')
        return rt

    def sawyer_to_franka_position(pos):
        # Convert from sawyer code representation of X,Y to Franka
        pos[0] *= -1 # The x is oposite sign from the sawyer code
        pos[:2] = pos[:2][::-1] # The x and y are switched compared to sawyer for which code was written
        return pos
    
    def go_to_cartesian_pose(self, positions, orientations, precise=False):
        """
            Move to a list of points in space
            args:
                positions (np.array(n,3)) : x,y,z coordinates in meters from robot origin
                orientations (np.array(n,4)) : x,y,z,w quaternion orientation
                precise (bool) : use precise for slow short movements. else use False, which is fast but unstable
        """
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None,:]
            orientations = orientations[None,:]

        if precise:
            self.go_to_cartesian_pose_precise(positions, orientations)
        else:
            self.go_to_cartesian_pose_stable(positions, orientations)


    def go_to_cartesian_pose_stable(self, positions, orientations):
        for i in range(len(positions)):
            pos = Franka.sawyer_to_franka_position(positions[i])
            rt = Franka.create_rotation_transform(pos, orientations[i])

            # Determine speed/duration
            curr_pos = self.fa.get_pose().translation
            # print(curr_pos, pos)
            dist = ((curr_pos - pos)**2).sum()**.5
            # print('distance', dist)
            duration = dist * 7 # 1cm=.1s 1m=10s
            duration = max(0.6, duration) # Don't go toooo fast
            # print('duration', duration, type(duration))
            duration = float(duration)
            if pos[2] < 0.05:
                print('below threshold!!', pos[2])
                continue
            try:
                self.fa.goto_pose(rt,
                        duration=duration, 
                        force_thresholds=[10,10,10,10,10,10],
                        ignore_virtual_walls=True
                )
            except Exception as e:
                print('Could not goto_pose', e)
    
    def go_to_cartesian_pose_precise(self, positions, orientations, hertz=200, stiffness_factor=3.0):
        """
            This is a smooth version of this function. It can very smoothly go betwen the positions.
            However, it is unstable, and will result in oscilations sometimes.
            Recommended to be used only for fine, slow motions like the actual brush strokes.
        """
        from frankapy import SensorDataMessageType
        from frankapy import FrankaConstants as FC
        from frankapy.proto_utils import sensor_proto2ros_msg, make_sensor_group_msg
        from frankapy.proto import PosePositionSensorMessage, CartesianImpedanceSensorMessage
        from franka_interface_msgs.msg import SensorDataGroup
        from frankapy.utils import min_jerk, min_jerk_weight
        import rospy
        
        def get_duration(here, there):
            dist = ((here.translation - there.translation)**2).sum()**.5
            duration = dist *  10#5 # 1cm=.1s 1m=10s
            duration = max(0.2, duration) # Don't go toooo fast
            duration = float(duration)
            return duration, dist

        def smooth_trajectory(poses, window_width=50):
            # x = np.cumsum(delta_ts)
            from scipy.interpolate import interp1d
            from scipy.ndimage import gaussian_filter1d

            for c in range(3):
                coords = np.array([p.translation[c] for p in poses])

                coords_smooth = gaussian_filter1d(coords, 31)
                print(len(poses), len(coords_smooth))
                for i in range(len(poses)-1):
                    coords_smooth[i]
                    poses[i].translation[c] = coords_smooth[i]
            return poses

        pose_trajs = []
        delta_ts = []
        
        # Loop through each position/orientation and create interpolations between the points
        p0 = self.fa.get_pose()
        for i in range(len(positions)):
            p1 = Franka.create_rotation_transform(\
                Franka.sawyer_to_franka_position(positions[i]), orientations[i])

            duration, distance = get_duration(p0, p1)

            # needs to be high to avoid torque discontinuity error controller_torque_discontinuity
            STEPS = max(10, int(duration*hertz))
            # print(STEPS, distance)

            # if distance*100 > 5:
            #     print("You're using the precise movement wrong", distance*100)

            ts = np.arange(0, duration, duration/STEPS)
            # ts = np.linspace(0, duration, STEPS)
            weights = [min_jerk_weight(t, duration) for t in ts]

            if i == 0 or i == len(positions)-1:
                # Smooth for the first and last way points
                pose_traj = [p0.interpolate_with(p1, w) for w in weights]
            else:
                # linear for middle points cuz it's fast and accurate
                pose_traj = p0.linear_trajectory_to(p1, STEPS)
            # pose_traj = [p0.interpolate_with(p1, w) for w in weights]
            # pose_traj = p0.linear_trajectory_to(p1, STEPS)

            pose_trajs += pose_traj
            
            delta_ts += [duration/len(pose_traj),]*len(pose_traj)
            
            p0 = p1
            
        T = float(np.array(delta_ts).sum())

        # pose_trajs = smooth_trajectory(pose_trajs)

        pub = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC, SensorDataGroup, queue_size=1000)
        
        # To ensure skill doesn't end before completing trajectory, make the buffer time much longer than needed
        self.fa.goto_pose(pose_trajs[1], duration=T, dynamic=True, 
            buffer_time=T+10,
            force_thresholds=[10,10,10,10,10,10],
            cartesian_impedances=(np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES)*stiffness_factor).tolist() + FC.DEFAULT_ROTATIONAL_STIFFNESSES,
            ignore_virtual_walls=True,
        )
        try:
            init_time = rospy.Time.now().to_time()
            for i in range(2, len(pose_trajs)):
                timestamp = rospy.Time.now().to_time() - init_time
                traj_gen_proto_msg = PosePositionSensorMessage(
                    id=i, timestamp=timestamp, 
                    position=pose_trajs[i].translation, quaternion=pose_trajs[i].quaternion
                )
                fb_ctrlr_proto = CartesianImpedanceSensorMessage(
                    id=i, timestamp=timestamp,
                    # translational_stiffnesses=FC.DEFAULT_TRANSLATIONAL_STIFFNESSES[:2] + [z_stiffness_trajs[i]],
                    translational_stiffnesses=(np.array(FC.DEFAULT_TRANSLATIONAL_STIFFNESSES)*stiffness_factor).tolist(),
                    rotational_stiffnesses=FC.DEFAULT_ROTATIONAL_STIFFNESSES
                )
                
                ros_msg = make_sensor_group_msg(
                    trajectory_generator_sensor_msg=sensor_proto2ros_msg(
                        traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
                    feedback_controller_sensor_msg=sensor_proto2ros_msg(
                        fb_ctrlr_proto, SensorDataMessageType.CARTESIAN_IMPEDANCE)
                )

                # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
                pub.publish(ros_msg)
                # rate = rospy.Rate(1 / (delta_ts[i]))
                rate = rospy.Rate(hertz)
                rate.sleep()

                # if i%100==0:
                #     print(self.fa.get_pose().translation[-1] - pose_trajs[i].translation[-1], 
                #         '\t', self.fa.get_pose().translation[-1], '\t', pose_trajs[i].translation[-1])
        except Exception as e:
            print('unable to execute skill', e)
        # Stop the skill
        self.fa.stop_skill()
        
class SimulatedRobot(Robot, object):
    def __init__(self, debug=True):
        pass

    def good_morning_robot(self):
        pass

    def good_night_robot(self):
        pass

    def go_to_cartesian_pose(self, position, orientation, precise):
        pass


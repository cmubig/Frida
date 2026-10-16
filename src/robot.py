#! /usr/bin/env python
import os
import numpy as np
import time

from brush_stroke import euler_from_quaternion

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

class XArm(Robot, object):
    '''
        Low-level action functionality of the robot.
        This is an abstract class, see its children for usage.
    '''
    ROLL, PITCH, YAW = 180.0, 0.0, 0.0

    def __init__(self, ip, debug, tool_length_m=0.0, neutral_joints_deg=None,
            v_max_mm_s=60.0, a_lat_mm_s2=150.0, joint_rate_limit_deg_s=60.0):
        from xarm.wrapper import XArmAPI
        self.debug_bool = debug
        self.tool_length_m = float(tool_length_m or 0.0)
        self.neutral_joints_deg = None if neutral_joints_deg is None else [float(a) for a in neutral_joints_deg]

        # Contact motion profile (mm, mm/s, mm/s^2, deg/s)
        self.v_max = float(v_max_mm_s)
        self.v_min = 12.0
        self.v_air = 70.0
        self.a_lat = float(a_lat_mm_s2)
        # Keep mvacc moderate: high accel with a long TCP trips error 24 at corners.
        self.mvacc = 280.0
        self.joint_rate_limit = float(joint_rate_limit_deg_s)
        self.sharp_turn_deg = 25.0
        self.transit_min_mm = 30.0
        self.joint_speed_deg = 50.0
        self.joint_acc_deg = 350.0
        # How many waypoints to queue before waiting (stop-and-go, no blend).
        self.queue_chunk = 24
        self._ik_cache = {}

        self.arm = XArmAPI(ip)

    def debug(self, msg):
        if self.debug_bool:
            print(msg)

    def _apply_tcp_offset(self):
        '''Set TCP at the tool tip so cartesian poses are tip-referenced.

        xArm expects [x, y, z, roll, pitch, yaw] in mm / degrees.
        Does not change joint-space go-home; avoid reset()/move_gohome with long tools.
        '''
        tip_mm = self.tool_length_m * 1000.0
        self.debug(f'XArm TCP offset z={tip_mm:.1f} mm (TOOL_LENGTH_M={self.tool_length_m})')
        # Offset along tool +Z (pen pointing down toward the table).
        code = self.arm.set_tcp_offset([0, 0, tip_mm, 0, 0, 0])
        if code not in (0, None):
            print(f'Warning: set_tcp_offset returned {code}')

    def good_morning_robot(self):
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        # Apply TCP before any cartesian motion. Skip arm.reset()/move_gohome —
        # that joint home ignores TCP and can drive a long pen into the table.
        self._apply_tcp_offset()
        self.arm.set_state(state=0)
        self._ik_cache = {}
        try:
            print('XArm reduced mode:', self.arm.get_reduced_mode())
            print('XArm reduced states:', self.arm.get_reduced_states())
        except Exception as e:
            print('XArm could not read reduced mode:', e)

    def good_night_robot(self):
        self.arm.disconnect()

    # def go_to_cartesian_pose(self, positions, orientations,
    #         speed=250):
    #     # positions in meters
    #     positions, orientations = np.array(positions), np.array(orientations)
    #     if len(positions.shape) == 1:
    #         positions = positions[None,:]
    #         orientations = orientations[None,:]
    #     for i in range(len(positions)):
    #         x,y,z = positions[i][1], positions[i][0], positions[i][2]
    #         x,y,z = x*1000, y*-1000, z*1000 #m to mm
    #         q = orientations[i]
            
    #         euler= euler_from_quaternion(q[0], q[1], q[2], q[3])#quaternion.as_quat_array(orientations[i])
    #         roll, pitch, yaw = 180, 0, 0#euler[0], euler[1], euler[2]
    #         # https://github.com/xArm-Developer/xArm-Python-SDK/blob/0fd107977ee9e66b6841ea9108583398a01f227b/xarm/x3/xarm.py#L214
            
    #         wait = True 
    #         failure, state = self.arm.get_position()
    #         if not failure:
    #             curr_x, curr_y, curr_z = state[0], state[1], state[2]
    #             # print('curr', curr_y, y)
    #             dist = ((x-curr_x)**2 + (y-curr_y)**2 + (z-curr_z)**2)**0.5
    #             # print('dist', dist)
    #             # Dist in mm
    #             if dist < 5:
    #                 wait=False
    #                 speed=600
    #                 # print('less')

    #         try:
    #             r = self.arm.set_position(
    #                     x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
    #                     speed=speed, wait=wait
    #             )
    #             # print(r)
    #             if r:
    #                 print("failed to go to pose, resetting.")
    #                 self.arm.clean_error()
    #                 self.good_morning_robot()
    #                 self.arm.set_position(
    #                         x=x, y=y, z=z+5, roll=roll, pitch=pitch, yaw=yaw,
    #                         speed=speed, wait=True
    #                 )
    #                 self.arm.set_position(
    #                         x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
    #                         speed=speed, wait=True
    #                 )
    #         except Exception as e:
    #             self.good_morning_robot()
    #             print('Cannot go to position', e)

    def go_to_cartesian_pose(self, positions, orientations,
            speed=250, fast=False, motion=False):
        # fast = False
        if fast:
            return self.go_to_cartesian_pose_fast(positions, orientations)
        # positions in meters
        positions, orientations = np.array(positions), np.array(orientations)
        if len(positions.shape) == 1:
            positions = positions[None,:]
            orientations = orientations[None,:]
        for i in range(len(positions)):
            x,y,z = positions[i][1], positions[i][0], positions[i][2]
            x,y,z = x*1000, y*-1000, z*1000 #m to mm
            q = orientations[i]
            
            # print("not in fast mode")
            # print('x', x)
            # print('y', y)
            # print('z', z)
            if (motion):
                # print('in motion')
                roll, pitch, yaw = orientations[i][0], orientations[i][1], orientations[i][2]
            else:
                # euler= euler_from_quaternion(q[0], q[1], q[2], q[3])#quaternion.as_quat_array(orientations[i])
                roll, pitch, yaw = 180, 0, 0#euler[0], euler[1], euler[2]
                # https://github.com/xArm-Developer/xArm-Python-SDK/blob/0fd107977ee9e66b6841ea9108583398a01f227b/xarm/x3/xarm.py#L214
            
            wait = True 
            failure, state = self.arm.get_position()
            if not failure:
                curr_x, curr_y, curr_z = state[0], state[1], state[2]
                # print('curr', curr_y, y)
                dist = ((x-curr_x)**2 + (y-curr_y)**2 + (z-curr_z)**2)**0.5
                # print('dist', dist)
                # Dist in mm
                if dist < 5:
                    # wait=False
                    speed=1000
                    # print('less')

            # r = self.arm.set_position(
            #         x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            #         speed=speed, wait=wait, mvacc=5000
            # )
            try:
                r = self.arm.set_position(
                        x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                        speed=speed, wait=wait, mvacc=5000, radius=3 if motion else None
                )
                # print(r)
                if r:
                    print("failed to go to pose, resetting.")
                    # self.arm.clean_error()
                    # self.good_morning_robot()
                    self.arm.set_position(
                            x=x, y=y, z=z+5, roll=roll, pitch=pitch, yaw=yaw,
                            speed=speed, wait=True
                    )
                    self.arm.set_position(
                            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                            speed=speed, wait=True
                    )
            except Exception as e:
                self.good_morning_robot()
                print('Cannot go to position', e)
    
    def _arm_error_code(self):
        try:
            code, err = self.arm.get_err_warn_code()
            if code == 0:
                return int(err[0])
        except Exception:
            pass
        return -1

    def _motion_ok(self):
        '''True if the controller has no error and isn't stopped.

        Use this instead of SDK return codes: move_arc_lines returns None.
        '''
        try:
            return self.arm.error_code == 0 and self.arm.state < 4
        except Exception:
            return False

    def _recover_motion(self, retries=6, sleep_s=0.35):
        '''Clear stop/error and wait until the controller is actually ready.'''
        for i in range(retries):
            try:
                self.arm.clean_warn()
                self.arm.clean_error()
                self.arm.motion_enable(enable=True)
                self.arm.set_mode(0)
                self.arm.set_state(state=0)
                time.sleep(sleep_s)
                if self._motion_ok():
                    return True
                print(f'XArm recover: still error={self._arm_error_code()} (try {i+1}/{retries})')
            except Exception as e:
                print('XArm recover_motion failed:', e)
            time.sleep(sleep_s)
        return self._motion_ok()

    def _ready(self):
        return self._motion_ok() or self._recover_motion()

    def _current_xyz(self):
        code, state = self.arm.get_position()
        if code != 0:
            return None
        return np.array(state[:3], dtype=np.float64)

    def _ik(self, xyz_mm):
        '''Joint angles (deg) for a tip-down pose at xyz_mm, or None if unreachable.'''
        key = tuple(float(c) for c in np.round(np.asarray(xyz_mm, dtype=np.float64), 1))
        if key in self._ik_cache:
            return self._ik_cache[key]
        q = None
        try:
            code, angles = self.arm.get_inverse_kinematics(
                [key[0], key[1], key[2], self.ROLL, self.PITCH, self.YAW],
                input_is_radian=False, return_is_radian=False,
            )
            if code == 0 and angles:
                q = np.array(angles, dtype=np.float64)
        except Exception as e:
            print('XArm IK failed:', e)
        self._ik_cache[key] = q
        return q

    def _fail(self, what, target_mm=None):
        '''Log, recover once, lift straight up a little. Always returns False.'''
        err = self._arm_error_code()
        try:
            _, joints = self.arm.get_servo_angle()
        except Exception:
            joints = None
        tgt = None if target_mm is None else np.round(np.asarray(target_mm), 1).tolist()
        joints = None if joints is None else np.round(np.asarray(joints, dtype=np.float64), 1).tolist()
        print(f'XArm {what} failed: err={err} target_mm={tgt} joints_deg={joints}')
        if self._recover_motion():
            curr = self._current_xyz()
            if curr is not None:
                self.arm.set_position(
                    x=float(curr[0]), y=float(curr[1]), z=float(curr[2] + 20.0),
                    roll=self.ROLL, pitch=self.PITCH, yaw=self.YAW,
                    speed=10, mvacc=100, wait=True,
                )
                if not self._motion_ok():
                    self._recover_motion()
        return False

    def _joint_move(self, xyz_mm=None, q=None):
        '''Joint-space move to a tip-down pose (or given joints); can't cross a cartesian singularity.'''
        if q is None:
            q = self._ik(xyz_mm)
            if q is None:
                print(f'XArm IK has no solution for {np.round(xyz_mm, 1).tolist()}')
                return False
        if not self._ready():
            return self._fail('joint move (not ready)', xyz_mm)
        self.arm.set_servo_angle(
            angle=[float(a) for a in q], speed=self.joint_speed_deg,
            mvacc=self.joint_acc_deg, is_radian=False, wait=True,
        )
        if not self._motion_ok():
            return self._fail('joint move', xyz_mm)
        return True

    def _frida_to_xarm_mm(self, positions):
        '''FRIDA meters (N,3)/(3,) → xArm mm (N,3).'''
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions[None, :]
        xyz = np.empty((len(positions), 3), dtype=np.float64)
        xyz[:, 0] = positions[:, 1] * 1000.0
        xyz[:, 1] = positions[:, 0] * -1000.0
        xyz[:, 2] = positions[:, 2] * 1000.0
        return xyz

    def go_neutral(self, position_m, hover_m=None):
        '''Lift clear of the paper (cartesian), then joint-space move to neutral.

        args:
            position_m: FRIDA-meter neutral position (used via IK unless
                neutral_joints_deg is set)
            hover_m: FRIDA-meter z the tip must reach before the joint move
        '''
        if not self._ready():
            return self._fail('neutral (not ready)')
        target = self._frida_to_xarm_mm(position_m)[0]
        curr = self._current_xyz()
        if curr is not None and hover_m is not None:
            hover_z = float(hover_m) * 1000.0
            if curr[2] < hover_z - 1.0:
                self.arm.set_position(
                    x=float(curr[0]), y=float(curr[1]), z=hover_z,
                    roll=self.ROLL, pitch=self.PITCH, yaw=self.YAW,
                    speed=self.v_air / 2, mvacc=self.mvacc, wait=True,
                )
                if not self._motion_ok():
                    return self._fail('neutral lift', [curr[0], curr[1], hover_z])
        if self.neutral_joints_deg is not None:
            return self._joint_move(q=self.neutral_joints_deg)
        return self._joint_move(target)

    def _transit(self, target_mm):
        '''Get the tip to target_mm: short hops in cartesian, longer ones in joint space.'''
        curr = self._current_xyz()
        if curr is not None and np.linalg.norm(curr - target_mm) < self.transit_min_mm:
            self.arm.set_position(
                x=float(target_mm[0]), y=float(target_mm[1]), z=float(target_mm[2]),
                roll=self.ROLL, pitch=self.PITCH, yaw=self.YAW,
                speed=self.v_air, mvacc=self.mvacc, wait=True,
            )
            if not self._motion_ok():
                return self._fail('short transit', target_mm)
            return True
        return self._joint_move(target_mm)

    def _contact_profile(self, xyz):
        '''Per-waypoint speed (mm/s) for the segment ending at i.

        No blending: the old reliable path stopped at every point (the ×70
        duplicate hack). Blends of ~0.5 mm at tens of mm/s need thousands of
        mm/s² of lateral accel and trip error 24.

        Speed follows local curvature, v = sqrt(a_lat * R); sharp corners and
        vertical approach/lift use slower air/min speeds.
        '''
        n = len(xyz)
        R = np.full(n, np.inf)
        dev = np.zeros(n)
        for i in range(1, n - 1):
            a, b, c = xyz[i - 1], xyz[i], xyz[i + 1]
            ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
            area2 = np.linalg.norm(np.cross(b - a, c - a))
            if area2 > 1e-9:
                R[i] = ab * bc * ca / (2.0 * area2)
            if ab > 1e-6 and bc > 1e-6:
                cos = float(np.clip(np.dot(b - a, c - b) / (ab * bc), -1.0, 1.0))
                dev[i] = np.degrees(np.arccos(cos))

        v = np.zeros(n)
        for i in range(1, n):
            seg = xyz[i] - xyz[i - 1]
            seg_len = float(np.linalg.norm(seg))
            vertical = seg_len > 1e-6 and np.linalg.norm(seg[:2]) < 0.3 * seg_len
            if vertical:
                v[i] = self.v_air
            else:
                r_min = min(R[i - 1], R[i])
                v[i] = float(np.clip(np.sqrt(self.a_lat * r_min), self.v_min, self.v_max))
            # Slow into / out of a sharp corner (still stop-and-go; no blend).
            if i < n - 1 and dev[i] >= self.sharp_turn_deg:
                v[i] = self.v_min
            if i > 1 and dev[i - 1] >= self.sharp_turn_deg:
                v[i] = min(v[i], self.v_min)
        return v

    def _preflight(self, xyz, v):
        '''IK every waypoint; reject singular/unreachable strokes, else scale speed to joint-rate limit.'''
        qs = []
        for p in xyz:
            q = self._ik(p)
            if q is None:
                print(f'XArm preflight: no IK for {np.round(p, 1).tolist()}; skipping stroke')
                return None
            qs.append(q)
        axis = int(getattr(self.arm, 'axis', 6) or 6)
        peak_rate = 0.0
        for i in range(1, len(xyz)):
            dq = (qs[i][:axis] - qs[i - 1][:axis] + 180.0) % 360.0 - 180.0
            dq_max = float(np.max(np.abs(dq)))
            seg_len = float(np.linalg.norm(xyz[i] - xyz[i - 1]))
            if seg_len < 5.0 and dq_max > 15.0:
                print(f'XArm preflight: near-singular segment ({dq_max:.1f} deg over '
                      f'{seg_len:.1f} mm) at {np.round(xyz[i], 1).tolist()}; skipping stroke')
                return None
            if seg_len > 1e-6:
                peak_rate = max(peak_rate, dq_max * v[i] / seg_len)
        if peak_rate > self.joint_rate_limit:
            scale = self.joint_rate_limit / peak_rate
            self.debug(f'XArm preflight: scaling stroke speed by {scale:.2f} '
                       f'(peak joint rate {peak_rate:.0f} deg/s)')
            v = v * scale
        return v

    def _nearest_waypoint(self, xyz, curr):
        if curr is None or len(xyz) == 0:
            return 0
        d = np.linalg.norm(xyz - curr[None, :], axis=1)
        return int(np.argmin(d))

    def go_to_cartesian_pose_fast(self, positions, orientations, dedup_mm=1.5):
        '''
        Stroke motion: joint-space transit to the first waypoint, then stop-and-go
        cartesian moves (no blend) with curvature-limited speed. Returns True/False.

        args:
            positions: (N,3) or (3,) meters in FRIDA coords
            orientations: ignored (tip is always pointing down)
        '''
        positions = np.asarray(positions, dtype=np.float64)
        if positions.ndim == 1:
            positions = positions[None, :]
        if len(positions) == 0:
            return True
        if not self._ready():
            return self._fail('stroke (not ready)')

        xyz = self._frida_to_xarm_mm(positions)

        keep = [0]
        for i in range(1, len(xyz)):
            if np.linalg.norm(xyz[i] - xyz[keep[-1]]) >= dedup_mm:
                keep.append(i)
        if keep[-1] != len(xyz) - 1:
            keep.append(len(xyz) - 1)
        xyz = xyz[keep]

        if len(xyz) == 1:
            return self._transit(xyz[0])

        v = self._contact_profile(xyz)
        v = self._preflight(xyz, v)
        if v is None:
            return False
        if not self._transit(xyz[0]):
            return False

        last = len(xyz) - 1
        # Stop-and-go: radius=None (SDK → -1) never blends. Queue small chunks
        # and wait on the last of each so mid-path error 24 is attributed correctly.
        # (XArmAPI wrapper has no wait_move; set_position(wait=True) does it.)
        i = 1
        while i <= last:
            chunk_end = min(i + self.queue_chunk - 1, last)
            for j in range(i, chunk_end + 1):
                self.arm.set_position(
                    x=float(xyz[j, 0]), y=float(xyz[j, 1]), z=float(xyz[j, 2]),
                    roll=self.ROLL, pitch=self.PITCH, yaw=self.YAW,
                    speed=float(v[j]), mvacc=self.mvacc,
                    radius=None, wait=(j == chunk_end),
                )
            if not self._motion_ok():
                curr = self._current_xyz()
                near = self._nearest_waypoint(xyz, curr)
                return self._fail(
                    f'stroke near waypoint {near}/{last} (chunk {i}-{chunk_end})',
                    None if curr is None else curr,
                )
            i = chunk_end + 1
        return True

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
    
    def go_to_cartesian_pose(self, positions, orientations):
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

        # if precise:
        #     self.go_to_cartesian_pose_precise(positions, orientations, stiffness_factor=stiffness_factor)
        # else:
        #     self.go_to_cartesian_pose_stable(positions, orientations)
        self.go_to_cartesian_pose_stable(positions, orientations)


    def go_to_cartesian_pose_stable(self, positions, orientations):
        abs_diffs = []
        # start = time.time()
        for i in range(len(positions)):
            pos = Franka.sawyer_to_franka_position(positions[i])
            rt = Franka.create_rotation_transform(pos, orientations[i])

            # Determine speed/duration
            curr_pos = self.fa.get_pose().translation
            # print(curr_pos, pos)
            dist = ((curr_pos - pos)**2).sum()**.5 
            # print('distance', dist*100)
            duration = dist * 5 # 1cm=.1s 1m=10s
            duration = max(0.6, duration) # Don't go toooo fast
            if dist*100 < 0.8: # less than 0.8cm
                # print('very short')
                duration = 0.3
            # print('duration', duration, type(duration))
            duration = float(duration)
            if pos[2] < 0.05:
                print('below threshold!!', pos[2])
                continue
            try:
                self.fa.goto_pose(rt,
                        duration=duration, 
                        force_thresholds=[10,10,10,10,10,10],
                        ignore_virtual_walls=True,
                        buffer_time=0.0
                )    
            except Exception as e:
                print('Could not goto_pose', e)
            abs_diff = sum((self.fa.get_pose().translation-rt.translation)**2)**0.5 * 100
            # print(abs_diff, 'cm stable')
            abs_diffs.append(abs_diff)
        # print(max(abs_diffs), sum(abs_diffs)/len(abs_diffs), '\t', time.time() - start)
        if abs_diffs[-1] > 1:
            # Didn't get to the last position. Try again.
            print('Off from final position by', abs_diffs[-1], 'cm')
            self.fa.goto_pose(rt,
                        duration=duration+3, 
                        force_thresholds=[10,10,10,10,10,10],
                        ignore_virtual_walls=True,
                        buffer_time=0.0
                )
            abs_diff = sum((self.fa.get_pose().translation-rt.translation)**2)**0.5 * 100
            if abs_diff > 1:
                print('Failed to get to end of trajectory again. Resetting Joints')
                self.fa.reset_joints()

    def go_to_cartesian_pose_precise(self, positions, orientations, hertz=300, stiffness_factor=3.0):
        """
            This is a smooth version of this function. It can very smoothly go betwen the positions.
            However, it is unstable, and will result in oscilations sometimes.
            Recommended to be used only for fine, slow motions like the actual brush strokes.
        """
        start = time.time()
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
            duration = max(0.4, duration) # Don't go toooo fast
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
        abs_diffs = []
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
                if i%10 == 0:
                    abs_diff = sum((self.fa.get_pose().translation-pose_trajs[i].translation)**2)**0.5 * 100
                    # print(abs_diff, 'cm')
                    abs_diffs.append(abs_diff)
                    # print(self.fa.get_pose().translation[-1] - pose_trajs[i].translation[-1], 
                    #     '\t', self.fa.get_pose().translation[-1], '\t', pose_trajs[i].translation[-1])
        except Exception as e:
            print('unable to execute skill', e)
        print(max(abs_diffs), sum(abs_diffs)/len(abs_diffs), '\t', time.time() - start)
        # Stop the skill
        self.fa.stop_skill()
        
class SimulatedRobot(Robot, object):
    def __init__(self, debug=True):
        pass

    def good_morning_robot(self):
        pass

    def good_night_robot(self):
        pass

    def go_to_cartesian_pose(self, position, orientation, **kwargs):
        return True

    def go_neutral(self, position_m, hover_m=None):
        return True



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
        # print(self.limb)

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
            # 1/0

        rospy.on_shutdown(clean_shutdown)
        self.debug("Excecuting... ")

        # neutral_pose = rospy.get_param("named_poses/{0}/poses/neutral".format(self.name))
        # angles = dict(list(zip(self.joint_names(), neutral_pose)))
        # self.set_joint_position_speed(0.1)
        # self.move_to_joint_positions(angles, timeout)
        intera_interface.Limb(synchronous_pub=True).move_to_neutral(speed=.2)
        
        return rs

    def good_night_robot(self):
        import rospy
        """ Tuck it in, read it a story """
        rospy.signal_shutdown("Example finished.")
        self.debug("Done")

    def go_to_cartesian_pose(self, position, orientation):
        #if len(position)
        position, orientation = np.array(position), np.array(orientation)
        if len(position.shape) == 1:
            position = position[None,:]
            orientation = orientation[None,:]

        # import rospy
        # import argparse
        from intera_motion_interface import (
            MotionTrajectory,
            MotionWaypoint,
            MotionWaypointOptions
        )
        from intera_motion_msgs.msg import TrajectoryOptions
        from geometry_msgs.msg import PoseStamped
        import PyKDL
        # from tf_conversions import posemath
        # from intera_interface import Limb

        limb = self.limb#Limb()

        traj_options = TrajectoryOptions()
        traj_options.interpolation_type = TrajectoryOptions.CARTESIAN
        traj = MotionTrajectory(trajectory_options = traj_options, limb = limb)

        wpt_opts = MotionWaypointOptions(max_linear_speed=0.8*1.5,
                                         max_linear_accel=0.8*1.5,
                                         # joint_tolerances=0.05,
                                         corner_distance=0.005,
                                         max_rotational_speed=1.57,
                                         max_rotational_accel=1.57,
                                         max_joint_speed_ratio=1.0)

        for i in range(len(position)):
            waypoint = MotionWaypoint(options = wpt_opts.to_msg(), limb = limb)

            joint_names = limb.joint_names()

            endpoint_state = limb.tip_state("right_hand")
            pose = endpoint_state.pose

            pose.position.x = position[i,0]
            pose.position.y = position[i,1]
            pose.position.z = position[i,2]

            pose.orientation.x = orientation[i,0]
            pose.orientation.y = orientation[i,1]
            pose.orientation.z = orientation[i,2]
            pose.orientation.w = orientation[i,3]

            poseStamped = PoseStamped()
            poseStamped.pose = pose

            joint_angles = limb.joint_ordered_angles()
            waypoint.set_cartesian_pose(poseStamped, "right_hand", joint_angles)

            traj.append_waypoint(waypoint.to_msg())

        result = traj.send_trajectory(timeout=None)
        # print(result.result)
        success = result.result
        # if success != True:
        #     print(success)
        if not success:
            import time
            # print('sleeping')
            time.sleep(2)
            # print('done sleeping. now to neutral')
            # Go to neutral and try again
            limb.move_to_neutral(speed=.3)
            # print('done to neutral')
            result = traj.send_trajectory(timeout=None)
            # print('just tried to resend trajectory')
            if result.result:
                print('second attempt successful')
            else:
                print('failed second attempt')
            success = result.result
        return success

    def move_to_joint_positions(self, position, timeout=3, speed=0.1):
        """
        args:
            dict{'right_j0',float} - dictionary of joint to joint angle
        """
        # rate = rospy.Rate(100)
        #try:
        # print('Positions:', position)
        self.limb.set_joint_position_speed(speed=speed)
        self.limb.move_to_joint_positions(position, timeout=timeout,
                                     threshold=0.008726646)
        self.limb.set_joint_position_speed(speed=.1)
        # rate.sleep()
        # except Exception as e:
        #     print('Exception while moving robot:\n', e)
        #     import traceback
        #     import sys
        #     print(traceback.format_exc())


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
        self.display_image(os.path.join(str(ros_dir), 'src', 'frida.jpg'))

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
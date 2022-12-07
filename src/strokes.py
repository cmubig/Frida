#! /usr/bin/env python3

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################


import math
import copy
import numpy as np
import time
from paint_utils import canvas_to_global_coordinates
from options import Options

# from painter import *

'''
Trajectory is given by a cubic bezier curve.
For creating cubic bezier curves, try this tool:
    https://www.desmos.com/calculator/ebdtbxgbq0
The trajectory is 4 x,y,z coordinates.
X and Y are in meters from the origin point
Z is a proportion where 1=max push down and 0 is barely touching the canvas
    this is Painter.Z_MAX_CANVAS and Painter.Z_CANVAS respectively.
'''


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
        https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    # roll = (roll/360.)*2.*math.pi
    # pitch = (pitch/360.)*2.*math.pi
    # yaw = (yaw/360.)*2.*math.pi
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
 
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
 
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
 
    return roll_x, pitch_y, yaw_z # in radians

def spherical_to_quaternion(theta, phi):
    return np.cos(theta)*np.sin(phi), \
            np.sin(theta)*np.sin(phi), \
            np.cos(phi)

class Stroke(object):
    '''
        Abstract brush stroke class.
        All brush strokes must have a trajectory which defines the path of the stroke
        along with how hard to push the brush down.
    '''
    def __init__(self, trajectory=None):
        self.trajectory = trajectory
        pass
    def paint(self, painter, x_start, y_start, rotation, step_size=.005):
        # x_start, y_start in global coordinates. rotation in radians

        # Need to translate x,y a bit to be accurate according to camera
        if painter.H_coord is not None:
            # Translate the coordinates so they're similar. see coordinate_calibration
            sim_coords = np.array([x_start, y_start, 1.])
            real_coords = painter.H_coord.dot(sim_coords)
            x_start, y_start = real_coords[0]/real_coords[2], real_coords[1]/real_coords[2]

        z_range = np.abs(painter.Z_MAX_CANVAS - painter.Z_CANVAS)

        path = self.get_rotated_trajectory(rotation)
        # painter.hover_above(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS)
        painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.07, speed=0.4)
        painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.01, speed=0.1)
        p0 = path[0,0], path[0,1], path[0,2]
        p3 = None

        for i in range(1, len(path)-1, 3):
            p1 = path[i+0,0], path[i+0,1], path[i+0,2]
            p2 = path[i+1,0], path[i+1,1], path[i+1,2]
            p3 = path[i+2,0], path[i+2,1], path[i+2,2]

            stroke_length = ((p3[0]-p0[0])**2 + (p3[1] - p0[1])**2)**.5
            n = max(2, int(stroke_length/step_size))
            n=30 # TODO: something more than this? see previous line
            for t in np.linspace(0,1,n):
                x = (1-t)**3 * p0[0] \
                      + 3*(1-t)**2*t*p1[0] \
                      + 3*(1-t)*t**2*p2[0] \
                      + t**3*p3[0]
                y = (1-t)**3 * p0[1] \
                      + 3*(1-t)**2*t*p1[1] \
                      + 3*(1-t)*t**2*p2[1] \
                      + t**3*p3[1]
                if t < 0.333:
                    z = (1 - t/.333) * p0[2] + (t/.333)*p1[2]
                elif t < 0.666:
                    z = (1 - (t-.333)/.333) * p1[2] + ((t-.333)/.333)*p2[2]
                else:
                    z = (1 - (t-.666)/.333) * p2[2] + ((t-.666)/.333)*p3[2]

                z = painter.Z_CANVAS - z * z_range
                painter.move_to(x_start+x, y_start+y, z, method='direct', speed=0.02)
                time.sleep(0.02)
            p0 = p3
        painter.move_to(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS + 0.01, speed=0.1)
        painter.move_to(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS + 0.07, speed=0.3)
        # painter.hover_above(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS)

    def angled_paint(self, painter, x_start, y_start, rotation, step_size=.005):
        # x_start, y_start in global coordinates. rotation in radians

        smooth = True
        if smooth:
            all_positions = []
            all_orientations = []

        # Need to translate x,y a bit to be accurate according to camera
        if painter.H_coord is not None:
            # Translate the coordinates so they're similar. see coordinate_calibration
            sim_coords = np.array([x_start, y_start, 1.])
            real_coords = painter.H_coord.dot(sim_coords)
            x_start, y_start = real_coords[0]/real_coords[2], real_coords[1]/real_coords[2]

        z_range = np.abs(painter.Z_MAX_CANVAS - painter.Z_CANVAS)

        path = self.get_rotated_trajectory(rotation)
        
        if smooth:
            all_positions.append([x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.07])
            all_orientations.append(None)
            all_positions.append([x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.02])
            all_orientations.append(None)
        else:
            painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.07, speed=0.4)
            painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.02, speed=0.1)

        p0 = path[0,0], path[0,1], path[0,2]
        p3 = None

        alpha = self.trajectory[-1][-1] # Same alpha throughout stroke, for now.
        # print('alpha', alpha)

        for i in range(1, len(path)-1, 3):
            p1 = path[i+0,0], path[i+0,1], path[i+0,2]
            p2 = path[i+1,0], path[i+1,1], path[i+1,2]
            p3 = path[i+2,0], path[i+2,1], path[i+2,2]

            stroke_length = ((p3[0]-p0[0])**2 + (p3[1] - p0[1])**2)**.5
            n = max(2, int(stroke_length/step_size))
            n=5#30 # TODO: something more than this? see previous line
            for t in np.linspace(0,1,n):
                x = (1-t)**3 * p0[0] \
                      + 3*(1-t)**2*t*p1[0] \
                      + 3*(1-t)*t**2*p2[0] \
                      + t**3*p3[0]
                y = (1-t)**3 * p0[1] \
                      + 3*(1-t)**2*t*p1[1] \
                      + 3*(1-t)*t**2*p2[1] \
                      + t**3*p3[1]
                if t < 0.333:
                    z = (1 - t/.333) * p0[2] + (t/.333)*p1[2]
                elif t < 0.666:
                    z = (1 - (t-.333)/.333) * p1[2] + ((t-.333)/.333)*p2[2]
                else:
                    z = (1 - (t-.666)/.333) * p2[2] + ((t-.666)/.333)*p3[2]

                def deriv_cubic_bez(p0,p1,p2,p3,t):
                    return -3*(1-t)**2*p0 \
                            + 3*(1-t)**2*p1 \
                            - 6*t*(1-t)*p1 \
                            - 3*t**2*p2 \
                            + 6*t*(1-t)*p2 \
                            + 3*t**2*p3
                dx_dt = deriv_cubic_bez(p0[0], p1[0], p2[0], p3[0], t)
                dy_dt = deriv_cubic_bez(p0[1], p1[1], p2[1], p3[1], t)
                dy_dx = dy_dt / dx_dt
                curve_angle = np.arctan(dy_dx)
                # print('curve_angle', curve_angle)

                def rad_to_deg(rad):
                    return 1.0*rad/math.pi * 180
                def deg_to_rad(deg):
                    return 1.0*deg/180*math.pi

                theta_sphere = np.arctan2(dy_dt, dx_dt) + np.pi/2 # the pi makes it perpendicular to trajectory

                phi_sphere = alpha
                # print(theta_sphere, phi_sphere)
                roll = np.cos(theta_sphere)*np.sin(phi_sphere)
                # pitch =  np.pi - np.sin(theta_sphere)*np.cos(phi_sphere)*np.sin(phi_sphere)
                pitch =  np.pi - np.sin(theta_sphere)*np.sin(phi_sphere)
                yaw = deg_to_rad(270.) # Constant yaw
                q = get_quaternion_from_euler(roll,pitch,yaw)

                #brush_length = 0.095
                l = painter.opt.brush_length
                if l is None:
                    print("must specify --brush_length")
                r = l * np.sin(phi_sphere)
                dx = r * np.cos(theta_sphere)
                dy = r * np.sin(theta_sphere)
                dz = l - l * np.cos(phi_sphere)
                # print('dx dy', dx, dy)
                x += dx
                y += dy
                #z -= dz
                # print('dz', dz)
                new_z_range = z_range * np.abs(np.cos(phi_sphere))
                # print('z range', z_range, new_z_range)
                # print('dz', dz)
                # print('painter.Z_CANVAS', painter.Z_CANVAS)
                # print('painter.Z_MAX_CANVAS', painter.Z_MAX_CANVAS)

                z = painter.Z_CANVAS - z * new_z_range - dz #+ 0.07
                # print(x,y,z)

                if smooth:
                    if t == 0:
                        all_positions.append([x_start+x, y_start+y, z+0.02])
                        all_orientations.append(q)
                        all_positions.append([x_start+x, y_start+y, z+0.005])
                        all_orientations.append(q)
                    all_positions.append([x_start+x, y_start+y, z])
                    all_orientations.append(q)
                    if t == 1:
                        all_positions.append([x_start+x, y_start+y, z+0.01])
                        all_orientations.append(q)
                        all_positions.append([x_start+x, y_start+y, z+0.02])
                        all_orientations.append(q)
                else:
                    if t == 0:
                        painter.move_to(x_start+x, y_start+y, z+0.02, q=q, method='direct', speed=0.1)
                        painter.move_to(x_start+x, y_start+y, z+0.005, q=q, method='direct', speed=0.03)
                    painter.move_to(x_start+x, y_start+y, z, q=q, method='direct', speed=0.05)
                    if t == 1:
                        painter.move_to(x_start+x, y_start+y, z+0.01, q=q, method='direct', speed=0.03)
                        painter.move_to(x_start+x, y_start+y, z+0.02, q=q, method='direct', speed=0.1)
                # time.sleep(0.02)
            p0 = p3

        if smooth:
            all_positions.append([x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS + 0.02])
            all_orientations.append(None)
            all_positions.append([x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS + 0.07])
            all_orientations.append(None)
        else:
            painter.move_to(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS + 0.02, speed=0.1)
            painter.move_to(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS + 0.07, speed=0.3)
        # painter.hover_above(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS)

        if smooth:
            painter.move_to_trajectories(all_positions, all_orientations)

    def get_rotated_trajectory(self, rotation):
        # Rotation in radians
        ret = copy.deepcopy(self.trajectory)
        for i in range(len(ret)):
            ret[i][0] = math.cos(rotation) * self.trajectory[i][0] \
                     - math.sin(rotation) * self.trajectory[i][1]
            ret[i][1] = math.sin(rotation) * self.trajectory[i][0] \
                     + math.cos(rotation) * self.trajectory[i][1]
        ret = np.array(ret)
        return ret

class StrokeA(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [-0.001,0,0.1], # [x,y,z]
            [-0.001,0,.7],
            [.000,0,.7],
            [.001,0,0.1]
        ]
class StrokeB(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.1],
            [.005,0,.1],
            [.01,0,.1],
            [.02,0,0.1]
        ]

class StrokeBA(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.1],
            [.005,0,.7],
            [.01,0,.7],
            [.02,0,0.1]
        ]

class StrokeBB(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.1],
            [.005,0.01,.3],
            [.01,0.01,.3],
            [.02,0,0.1]
        ]
class StrokeBC(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.1],
            [.01,0,.3],
            [.02,0,.3],
            [.03,0,0.1]
        ]
class StrokeBD(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.7],
            [.02,0.02,1.],
            [.03,-.02,1.],
            [.05,0,1.]
        ]

class StrokeC(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.3],
            [.008,.01,.5],
            [.016,0.01,.5],
            [.025,0,.5]
        ]
class StrokeD(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.5],
            [.01,-0.02,.6],
            [.02,-0.02,.5],
            [.04,0,0.3]
        ]
class StrokeE(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.4],
            [.01,0.02,.6],
            [.02,0.02,.5],
            [.04,0,0.3]
        ]
class StrokeF(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.1],
            [.006,0,.1],
            [.013,0,.1],
            [.02,0,0.1]
        ]
class StrokeG(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.7],
            [.02,0,1.],
            [.03,0,1.],
            [.05,0,1.]
        ]
class StrokeH(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.1],
            [.003,0,.2],
            [.006,0,.2],
            [.01,0,0.1]
        ]
class StrokeI(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.7],
            [.01,-0.01,1.],
            [.03,.01,1.],
            [.04,0,1.]
        ]

class StrokeJ(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.1],
            [.005,0.01,.1],
            [.01,0.01,.1],
            [.02,0,0.1]
        ]
class StrokeK(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.5],
            [.01,-0.01,.6],
            [.02,-0.01,.5],
            [.03,0,0.3]
        ]
class StrokeL(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.4],
            [.01,0.01,.6],
            [.02,0.01,.5],
            [.03,0,0.3]
        ]

class StrokeM(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.5],
            [.006,-0.005,.3],
            [.01,-0.005,.3],
            [.02,0,0.3]
        ]
class StrokeN(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.4],
            [.006,0.005,.3],
            [.012,0.005,.3],
            [.02,0,0.3]
        ]
        
class StrokeO(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.2],
            [.005,-0.01,.3],
            [.01,0.01,.3],
            [.015,0,0.1]
        ]
class StrokeP(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.2],
            [.005,0.01,.3],
            [.01,-0.01,.3],
            [.015,0,0.1]
        ]
class StrokeQ(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.1],
            [.002,0.001,.2],
            [.004,0.001,.2],
            [.005,0,0.1]
        ]
all_strokes = sorted(Stroke.__subclasses__(), key=lambda x : x.__class__.__name__)

def get_base_strokes():

    strokes = []

    # Dabs
    for z in np.linspace(0, 1, 3, endpoint=True):
        strokes.append(Stroke(trajectory=[
                [0,0,z],
                [.001,0,z],
                [.001,0,z],
                [.002,0,z]
            ]))

    # Very Short Strokes
    for z in np.linspace(0, 1, 3, endpoint=True):
        strokes.append(Stroke(trajectory=[
                [0,0,z],
                [.003,0,z],
                [.007,0,1-z],
                [.01,0,1-z]
            ]))
    for z in np.linspace(0, 1, 3, endpoint=True):
        strokes.append(Stroke(trajectory=[
                [0,0,1-z],
                [.003,0,1-z],
                [.007,0,z],
                [.01,0,z]
            ]))

    # Short-Medium Curved Strokes
    for z in np.linspace(0.3, .8, 2, endpoint=True):
        for y in np.linspace(-.02, 0.02, 4, endpoint=True):
            strokes.append(Stroke(trajectory=[
                    [0,0,z],
                    [.01,y,z],
                    [.015,y,z],
                    [.025,0,z]
                ]))
            strokes.append(Stroke(trajectory=[
                    [0,0,z],
                    [.005,y,z],
                    [.01,y,1-z],
                    [.02,0,1-z]
                ]))
            strokes.append(Stroke(trajectory=[
                    [0,0,z],
                    [.02,y,z],
                    [.03,y,1-z],
                    [.04,0,1-z]
                ]))

    # Long Strokes
    for z in np.linspace(0.3, .8, 2, endpoint=True):
        for y0 in np.linspace(-.03, 0.03, 4, endpoint=True):
            for y1 in np.linspace(-.03, 0.03, 4, endpoint=True):
                if y0 == y1: continue

                strokes.append(Stroke(trajectory=[
                        [0,0,z],
                        [.02,y0,z],
                        [.04,y1,1-z],
                        [.05,0,1-z]
                    ]))
                # strokes.append(Stroke(trajectory=[
                #         [0,0,z],
                #         [.02,y0,z],
                #         [.04,y1,z],
                #         [.06,0,z]
                #     ]))
    print(len(strokes))
    return strokes

def simple_parameterization_to_real(stroke_length, bend, z, alpha=0):
    xs = (np.arange(4)/3.) * stroke_length

    stroke = Stroke(trajectory=[
        [xs[0], 0, .2, alpha],
        [xs[1], bend, z, alpha],
        [xs[2], bend, z, alpha],
        [xs[3], 0, .2, alpha],
    ])
    return stroke


def get_random_stroke(max_length=0.05, min_length=0.01, max_alpha=Options().MAX_ALPHA):
    stroke_length = np.random.rand(1)[0]*(max_length-min_length) + min_length
    
    bend = np.random.rand(1)[0]*.04 - .02 
    bend = min(bend, stroke_length) if bend > 0 else max(bend, -1*stroke_length)


    alpha = (np.random.rand(1)[0]*2-1) * max_alpha
    
    z = np.random.rand(1)[0]

    return simple_parameterization_to_real(stroke_length, bend, z, alpha)

# def get_random_stroke(max_length=0.05, min_length=0.015):
#     stroke_length = np.random.rand(1)*(max_length-min_length) + min_length
#     xs = (np.arange(4)/3.) * stroke_length

#     y0 = 0
#     y1_2 = np.random.rand(2)*.04 - .02 
#     y1, y2 = y1_2[0], y1_2[1]
#     y1 = min(y1, stroke_length) if y1 > 0 else max(y1, -1*stroke_length)
#     y2 = min(y2, stroke_length) if y2 > 0 else max(y2, -1*stroke_length)
#     y3 = 0 # end at zero cuz otherwise it's just a rotated stroke

#     # zs = np.random.rand(4)
#     z0, z3 = np.random.rand(1)[0], np.random.rand(1)[0]
#     z1, z2 = z0 + ((z3-z0)/3), z3 - ((z3-z0)/3)

#     stroke = Stroke(trajectory=[
#         [xs[0], y0, z0],
#         [xs[1], y1_2[0], z1],
#         [xs[2], y1_2[1], z2],
#         [xs[3], y3, z3],
#     ])
#     return stroke
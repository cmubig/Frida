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

from painter import *

'''
Trajectory is given by a cubic bezier curve.
For creating cubic bezier curves, try this tool:
    https://www.desmos.com/calculator/ebdtbxgbq0
The trajectory is 4 x,y,z coordinates.
X and Y are in meters from the origin point
Z is a proportion where 1=max push down and 0 is barely touching the canvas
    this is Painter.Z_MAX_CANVAS and Painter.Z_CANVAS respectively.
'''

class Stroke(object):
    '''
        Abstract brush stroke class.
        All brush strokes must have a trajectory which defines the path of the stroke
        along with how hard to push the brush down.
    '''
    def __init__(self):
        pass
    def paint(self, painter, x_start, y_start, rotation, step_size=.005):
        # x_start, y_start in global coordinates. rotation in radians
        z_range = np.abs(painter.Z_MAX_CANVAS - painter.Z_CANVAS)

        path = self.get_rotated_trajectory(rotation)
        painter.hover_above(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS)
        painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.02, speed=0.2)
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
                    z = ((1 - t/.333) * p0[2] + (t/.333)*p1[2]) / 2
                elif t < 0.666:
                    z = ((1 - (t-.333)/.333) * p1[2] + ((t-.333)/.333)*p2[2]) / 2
                else:
                    z = ((1 - (t-.666)/.333) * p2[2] + ((t-.666)/.333)*p3[2]) / 2
                z = painter.Z_CANVAS - z * z_range
                painter.move_to(x_start+x, y_start+y, z, method='direct', speed=0.03)
            p0 = p3

        painter.move_to(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS + 0.02, speed=0.2)
        painter.hover_above(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS)

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
            [-0.002,0,0.4], # [x,y,z]
            [-0.001,0,.7],
            [.000,0,.7],
            [.002,0,0.4]
        ]
class StrokeB(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.3],
            [.003,0,.3],
            [.006,0,.3],
            [.01,0,0.3]
        ]
class StrokeC(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.3],
            [.008,0,.5],
            [.016,0,.5],
            [.025,0,.5]
        ]
class StrokeD(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.6],
            [.01,-0.02,.6],
            [.02,-0.02,.3],
            [.03,0,0.1]
        ]
class StrokeE(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.6],
            [.01,0.02,.6],
            [.02,0.02,.3],
            [.03,0,0.1]
        ]
class StrokeF(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.5],
            [.02,0,.3],
            [.03,0,.3],
            [.04,0,0.3]
        ]
class StrokeG(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.5],
            [.02,0,.8],
            [.03,0,.9],
            [.04,0,0.7]
        ]
class StrokeH(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.5],
            [.01,0,.8],
            [.02,0,1.],
            [.03,0,0.8]
        ]
class StrokeI(Stroke):
    def __init__(self):
        super(Stroke, self).__init__()
        self.trajectory = [
            [0,0,0.5],
            [.005,0,.8],
            [.01,0,1.],
            [.015,0,0.8]
        ]

all_strokes = sorted(Stroke.__subclasses__(), key=lambda x : x.__class__.__name__)




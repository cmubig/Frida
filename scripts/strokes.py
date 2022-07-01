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
                    z = (1 - t/.333) * p0[2] + (t/.333)*p1[2]
                elif t < 0.666:
                    z = (1 - (t-.333)/.333) * p1[2] + ((t-.333)/.333)*p2[2]
                else:
                    z = (1 - (t-.666)/.333) * p2[2] + ((t-.666)/.333)*p3[2]

                z = painter.Z_CANVAS - z * z_range
                painter.move_to(x_start+x, y_start+y, z, method='direct', speed=0.03)
                time.sleep(0.02)
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
            [-0.002,0,0.1], # [x,y,z]
            [-0.001,0,.3],
            [.000,0,.3],
            [.002,0,0.1]
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
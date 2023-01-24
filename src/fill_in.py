#! /usr/bin/env python3

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################


import math
import numpy as np

from strokes import Stroke
from options import Options


opt = Options()
opt.gather_options()

'''
Trajectory is an oscilating brush stroke with desired height and length
'''

class FillIn(object):
    '''
        Class containing info for a fill-in or block-in operation with the robot
        It's a series of oscilating brush strokes with a certain amplitude and length
    '''
    

    def __init__(self, h0, h1, length, frequency=0.007, alpha=math.pi/12):
        '''
        args:
            h0 float : initial height of the fill-in in meters
            h1 float : final height of the fill-in in meters
            length float : length in meters of the stroke
        kwargs:
            frequency float meters : how spaced out the curves are in meters
            alpha  float radians : # how much to angle the brush in the direction of the fill in
        '''
        
        self.h0 = h0
        self.h1 = h1
        self.length = length
        self.z = 0.8 # Constant pressure of brush

        n_strokes = max(1, 2*int(math.ceil(length/frequency)))
        n_control_points = n_strokes*3 + 1
        self.trajectory = np.zeros([n_control_points, 4])

        # x values
        self.trajectory[:,0] = np.linspace(0, self.length, self.trajectory.shape[0])

        # y values
        self.trajectory[np.arange(n_control_points)%3==0,1] = 0 # ocilation always starts and ends at 0
        # peak values are a linear interpolation between the two end point heights
        peak_vals = np.linspace(self.h0, self.h1, (np.arange(n_control_points)%3!=0).sum())
        peak_vals[np.floor(np.arange(len(peak_vals))/2)%2!=0] *= -1 # Oscilate
        self.trajectory[np.arange(n_control_points)%3!=0,1] = peak_vals # peaks

        # z values
        self.trajectory[:,2] = self.z

        # alpha values (angle of the brush)
        self.trajectory[:,3] = alpha

        self.stroke = Stroke(trajectory = self.trajectory)


    def angled_paint(self, painter, x_start, y_start, rotation, step_size=.005):
        self.stroke.angled_paint(painter, x_start, y_start, rotation, step_size=step_size, curve_angle_is_rotation=True)



def get_random_fill_in():
    stroke_length = np.random.rand(1)[0]*(opt.MAX_FILL_IN_LENGTH-opt.MIN_FILL_IN_LENGTH) + opt.MIN_FILL_IN_LENGTH
    
    h0 = np.random.rand(1)[0]*(opt.MAX_FILL_IN_HEIGHT-opt.MIN_FILL_IN_HEIGHT) + opt.MIN_FILL_IN_HEIGHT
    h1 = np.random.rand(1)[0]*(opt.MAX_FILL_IN_HEIGHT-opt.MIN_FILL_IN_HEIGHT) + opt.MIN_FILL_IN_HEIGHT


    return FillIn(h0, h1, stroke_length)

#! /usr/bin/env python

import os
import time
import sys
# import rospy
import numpy as np
from tqdm import tqdm
import scipy.special

from paint_utils import *
from robot import *
from painting_materials import *

q = np.array([0.704020578925, 0.710172716916,0.00244101361829,0.00194372088834])


TABLE_Z = -0.099

CANVAS_POSITION = (0,.5,TABLE_Z)
CANVAS_WIDTH = .2
CANVAS_HEIGHT = .2

WATER_POSITION = (-.3,.6,TABLE_Z)
RAG_POSTITION = (-.3,.3,TABLE_Z)

PALLETTE_POSITION = (.3,.5,TABLE_Z)
PAINT_DIFFERENCE = 0.03976

GET_PAINT_FREQ = 2

HOVER_FACTOR = 0.1


class Painter():

    def __init__(self, robot="sawyer"):
        self.robot = None
        if robot == "sawyer":
            self.robot = Sawyer(debug=True)

        self.robot.good_morning_robot()

        self.curr_position = None

        self.to_neutral()

    def to_neutral(self):
        # Initial spot
        self._move(0,0.5,TABLE_Z+0.1, timeout=20, method="direct")

    def _move(self, x, y, z, timeout=20, method='linear', step_size=.2, speed=0.1):
        '''
        Move to given x, y, z in global coordinates
        kargs:
            method 'linear'|'curved'|'direct'
        '''
        if self.curr_position is None:
            self.curr_position = [x, y, z]

        # Calculate how many
        dist = ((x-curr_position[0])**2 + (y-curr_position[1])**2 + (z-curr_position[2])**2)**(0.5)
        n_steps = max(2, int(dist//step_size))

        if method == 'linear':
            x_s = np.linspace(curr_position[0], x, n_steps)
            y_s = np.linspace(curr_position[1], y, n_steps)
            z_s = np.linspace(curr_position[2], z, n_steps)

            for i in range(1,n_steps):
                pos = self.robot.inverse_kinematics([x_s[i], y_s[i], z_s[i]], q)

                try:
                    self.robot.move_to_joint_positions(pos, timeout=timeout, speed=speed)
                except Exception as e:
                    print("error moving robot: ", e)
        elif method == 'curved':
            # TODO
            pass
        else:
            # Direct
            pos = self.robot.inverse_kinematics([x, y, z], q)
            self.robot.move_to_joint_positions(pos, timeout=timeout, speed=speed)

        self.curr_position = [x, y, z]

    def hover_above(self, x,y,z, method='linear'):
        self._move(x,y,z+HOVER_FACTOR, method=method, speed=0.4)
        # rate = rospy.Rate(100)
        # rate.sleep()

    def move_to(self, x,y,z, method='linear', speed=0.05):
        self._move(x,y,z, method=method, speed=speed)

    def dip_brush_in_water(self):
        self.hover_above(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2])
        self.move_to(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2], speed=0.2)
        rate = rospy.Rate(100)
        for i in range(5):
            noise = np.clip(np.random.randn(2)*0.01, a_min=-.02, a_max=0.02)
            self.move_to(WATER_POSITION[0]+noise[0],WATER_POSITION[1]+noise[1],WATER_POSITION[2], method='direct')
            rate.sleep()
        self.hover_above(WATER_POSITION[0],WATER_POSITION[1],WATER_POSITION[2])

    def rub_brush_on_rag(self):
        self.hover_above(RAG_POSTITION[0],RAG_POSTITION[1],RAG_POSTITION[2])
        self.move_to(RAG_POSTITION[0],RAG_POSTITION[1],RAG_POSTITION[2], speed=0.2)
        for i in range(5):
            noise = np.clip(np.random.randn(2)*0.02, a_min=-.03, a_max=0.03)
            self.move_to(RAG_POSTITION[0]+noise[0],RAG_POSTITION[1]+noise[1],RAG_POSTITION[2], method='direct')
        self.hover_above(RAG_POSTITION[0],RAG_POSTITION[1],RAG_POSTITION[2])

    def clean_paint_brush(self):
        self.dip_brush_in_water()
        self.rub_brush_on_rag()

    def get_paint(self, paint_index):
        x_offset = PAINT_DIFFERENCE * np.floor(paint_index/6)
        y_offset = PAINT_DIFFERENCE * (paint_index%6)

        x = PALLETTE_POSITION[0] + x_offset
        y = PALLETTE_POSITION[1] + y_offset
        z = PALLETTE_POSITION[2]

        self.hover_above(x,y,z)
        self.move_to(x,y,z + 0.02, speed=0.2)
        for i in range(3):
            noise = np.clip(np.random.randn(2)*0.0025, a_min=-.005, a_max=0.005)
            self.move_to(x+noise[0],y+noise[1],z, method='direct')
        self.move_to(x,y,z + 0.02, speed=0.2)
        self.hover_above(x,y,z)

    def paint_cubic_bezier(self, path, step_size=.005):
        """
        Paint 1 or more cubic bezier curves.
        Path is k*3+1 points, where k is # of bezier curves
        args:
            path np.array([n,2]) : x,y coordinates of a path of a brush stroke
        """

        p0 = canvas_to_global_coordinates(path[0,0], path[0,1], TABLE_Z)
        self.hover_above(p0[0], p0[1], TABLE_Z)
        self.move_to(p0[0], p0[1], TABLE_Z + 0.02, speed=0.2)
        p3 = None

        for i in range(1, len(path)-1, 3):
            p1 = canvas_to_global_coordinates(path[i+0,0], path[i+0,1], TABLE_Z)
            p2 = canvas_to_global_coordinates(path[i+1,0], path[i+1,1], TABLE_Z)
            p3 = canvas_to_global_coordinates(path[i+2,0], path[i+2,1], TABLE_Z)

            stroke_length = ((p3[0]-p0[0])**2 + (p3[1] - p0[1])**2)**.5
            n = max(2, int(stroke_length/step_size))
            n=10
            for t in np.linspace(0,1,n):
                x = (1-t)**3 * p0[0] \
                      + 3*(1-t)**2*t*p1[0] \
                      + 3*(1-t)*t**2*p2[0] \
                      + t**3*p3[0]
                y = (1-t)**3 * p0[1] \
                      + 3*(1-t)**2*t*p1[1] \
                      + 3*(1-t)*t**2*p2[1] \
                      + t**3*p3[1]
                self.move_to(x, y, TABLE_Z, method='direct', speed=0.03)
            p0 = p3

        pn = canvas_to_global_coordinates(path[-1,0], path[-1,1], TABLE_Z)
        self.move_to(pn[0], pn[1], TABLE_Z + 0.02, speed=0.2)
        self.hover_above(pn[0], pn[1], TABLE_Z)


    def paint_quadratic_bezier(self, p0,p1,p2, step_size=.005):
        p0 = canvas_to_global_coordinates(p0[0], p0[1], TABLE_Z)
        p1 = canvas_to_global_coordinates(p1[0], p1[1], TABLE_Z)
        p2 = canvas_to_global_coordinates(p2[0], p2[1], TABLE_Z)

        stroke_length = ((p1[0]-p0[0])**2 + (p1[1] - p0[1])**2)**.5 \
                + ((p2[0]-p1[0])**2 + (p2[1] - p1[1])**2)**.5
        # print('stroke_length', stroke_length)
        n = max(2, int(stroke_length/step_size))
        # print('n',n)

        self.hover_above(p0[0], p0[1], TABLE_Z)
        self.move_to(p0[0], p0[1], TABLE_Z + 0.02, speed=0.2)
        for t in np.linspace(0,1,n):
            x = (1-t)**2*p0[0] + 2*(1-t)*t*p1[0] + t**2*p2[0]
            y = (1-t)**2*p0[1] + 2*(1-t)*t*p1[1] + t**2*p2[1]
            self.move_to(x,y,TABLE_Z, method='direct')
        self.hover_above(p2[0],p2[1],TABLE_Z)

    def set_brush_height(self):
        # set the robot arm at a location on the canvas and
        # wait for the user to attach the brush

        p = canvas_to_global_coordinates(.5, .5, TABLE_Z)
        self.hover_above(p[0],p[1],TABLE_Z)
        self.move_to(p[0],p[1],TABLE_Z, method='direct')

        raw_input('Attach the paint brush now. Press enter to continue:')

        self.hover_above(p[0],p[1],TABLE_Z)
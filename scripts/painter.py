#! /usr/bin/env python

import os
import time
import sys
import numpy as np
from tqdm import tqdm
import scipy.special
import pickle

from paint_utils import *
from robot import *
from painting_materials import *
from strokes import paint_stroke_library
from simulated_painting_environment import pick_next_stroke

q = np.array([0.704020578925, 0.710172716916,0.00244101361829,0.00194372088834])
# q = np.array([0.1,0.2,0.3])
# q = np.array([.9,.155,.127,.05])

INIT_TABLE_Z = 0.

# Dimensions of canvas in meters
# CANVAS_WIDTH  = 0.3047 # 12"
# CANVAS_HEIGHT = 0.2285 # 9"
CANVAS_WIDTH  = 0.254 -0.005# 10"
CANVAS_HEIGHT = 0.2032 -0.005# 8"

# X,Y of canvas wrt to robot center (global coordinates)
CANVAS_POSITION = (0,.5) 

""" How many times in a row can you paint with the same color before needing more paint """
GET_PAINT_FREQ = 3

HOVER_FACTOR = 0.1


# Number of cells to paint in x and y directions
cells_x, cells_y = 3, 4

# Dimensions of the cells in Meters
#cell_dim = (0.0254, 0.0508) #h/w in meters. 1"x2"
cell_dim_y, cell_dim_x = CANVAS_HEIGHT / cells_y, CANVAS_WIDTH / cells_x

# The brush stroke starts halfway down and 20% over from left edge of cell
down = 0.5 * cell_dim_y
over = 0.2 * cell_dim_x

from stroke_calibration import process_stroke_library

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

class Painter():

    def __init__(self, robot="sawyer", use_cache=False, camera=None):

        self.robot = None
        if robot == "sawyer":
            self.robot = Sawyer(debug=True)

        self.robot.good_morning_robot()

        self.curr_position = None
        self.seed_position = None

        self.GET_PAINT_FREQ = GET_PAINT_FREQ

        self.to_neutral()

        # Set how high the table is wrt the brush
        if use_cache:
            params = pickle.load(open("cached_params.pkl",'rb'))
            self.Z_CANVAS = params['Z_CANVAS']
            self.Z_MAX_CANVAS = params['Z_MAX_CANVAS']
        else:
            print('Brush should be at bottom left of canvas.')
            print('Use keys "w" and "s" to set the brush to just barely touch the canvas.')
            p = canvas_to_global_coordinates(0, 0, INIT_TABLE_Z)
            self.Z_CANVAS = self.set_height(p[0], p[1], INIT_TABLE_Z)[2]

            print('Moving brush tip to the top right of canvas.')
            p = canvas_to_global_coordinates(1, 1, INIT_TABLE_Z)
            self.hover_above(p[0], p[1], self.Z_CANVAS, method='direct')

            print('Move the brush to the lowest point it should go.')
            self.Z_MAX_CANVAS = self.set_height(p[0], p[1], self.Z_CANVAS)[2]

            params = {'Z_CANVAS':self.Z_CANVAS, 'Z_MAX_CANVAS':self.Z_MAX_CANVAS}
            with open('cached_params.pkl','wb') as f:
                pickle.dump(params, f)
            self.to_neutral()
        
        self.Z_RANGE = np.abs(self.Z_MAX_CANVAS - self.Z_CANVAS)

        self.WATER_POSITION = (-.4,.6,self.Z_CANVAS)
        self.RAG_POSTITION = (-.4,.3,self.Z_CANVAS)

        self.PALLETTE_POSITION = (-.3,.5,self.Z_CANVAS- 0.5*self.Z_RANGE)
        self.PAINT_DIFFERENCE = 0.03976

        # Setup Camera
        self.camera = camera
        if self.camera is not None:
            self.camera.debug = True
            self.camera.calibrate_canvas(use_cache=use_cache)
            # Color callibration
            # self.camera.get_color_correct_image()

        # Get brush strokes from stroke library
        if not os.path.exists('strokes.pkl') or not use_cache:
            try:
                input('Need to create stroke library. Press enter to start.')
            except SyntaxError:
                pass

            paint_stroke_library(self)
            self.to_neutral()
            self.strokes = process_stroke_library(self.camera.get_canvas())
            with open('strokes.pkl','wb') as f:
                pickle.dump(self.strokes, f)
        else:
            self.strokes = pickle.load(open("strokes.pkl",'rb'))


    def next_stroke(self, canvas, target, colors, x_y_attempts=5, weight=None, loss_fcn=lambda c,t: np.abs(c - t)):
        return pick_next_stroke(canvas, target, self.strokes, colors, 
                    x_y_attempts=x_y_attempts, weight=weight, loss_fcn=loss_fcn)


    def to_neutral(self):
        # Initial spot
        self._move(0.2,0.5,INIT_TABLE_Z+0.05, timeout=20, method="direct", speed=0.25)

    def _move(self, x, y, z, timeout=20, method='linear', step_size=.2, speed=0.1):
        '''
        Move to given x, y, z in global coordinates
        kargs:
            method 'linear'|'curved'|'direct'
        '''
        if self.curr_position is None:
            self.curr_position = [x, y, z]

        # Calculate how many
        dist = ((x-self.curr_position[0])**2 + (y-self.curr_position[1])**2 + (z-self.curr_position[2])**2)**(0.5)
        n_steps = max(2, int(dist//step_size))

        if method == 'linear':
            x_s = np.linspace(self.curr_position[0], x, n_steps)
            y_s = np.linspace(self.curr_position[1], y, n_steps)
            z_s = np.linspace(self.curr_position[2], z, n_steps)

            for i in range(1,n_steps):
                pos = self.robot.inverse_kinematics([x_s[i], y_s[i], z_s[i]], q, seed_position=self.seed_position)
                self.seed_position = pos
                try:
                    self.robot.move_to_joint_positions(pos, timeout=timeout, speed=speed)
                except Exception as e:
                    print("error moving robot: ", e)
        elif method == 'curved':
            # TODO
            pass
        else:
            # Direct
            pos = self.robot.inverse_kinematics([x, y, z], q, seed_position=self.seed_position)
            self.seed_position = pos
            self.robot.move_to_joint_positions(pos, timeout=timeout, speed=speed)

        self.curr_position = [x, y, z]

    def hover_above(self, x,y,z, method='linear'):
        self._move(x,y,z+HOVER_FACTOR, method=method, speed=0.25)
        # rate = rospy.Rate(100)
        # rate.sleep()

    def move_to(self, x,y,z, method='linear', speed=0.05):
        self._move(x,y,z, method=method, speed=speed)

    def dip_brush_in_water(self):
        self.hover_above(self.WATER_POSITION[0],self.WATER_POSITION[1],self.WATER_POSITION[2])
        self.move_to(self.WATER_POSITION[0],self.WATER_POSITION[1],self.WATER_POSITION[2], speed=0.2)
        rate = rospy.Rate(100)
        for i in range(5):
            noise = np.clip(np.random.randn(2)*0.01, a_min=-.02, a_max=0.02)
            self.move_to(self.WATER_POSITION[0]+noise[0],self.WATER_POSITION[1]+noise[1],self.WATER_POSITION[2], method='direct')
            rate.sleep()
        self.hover_above(self.WATER_POSITION[0],self.WATER_POSITION[1],self.WATER_POSITION[2])

    def rub_brush_on_rag(self):
        self.hover_above(self.RAG_POSTITION[0],self.RAG_POSTITION[1],self.RAG_POSTITION[2])
        self.move_to(self.RAG_POSTITION[0],self.RAG_POSTITION[1],self.RAG_POSTITION[2], speed=0.2)
        for i in range(5):
            noise = np.clip(np.random.randn(2)*0.02, a_min=-.03, a_max=0.03)
            self.move_to(self.RAG_POSTITION[0]+noise[0],self.RAG_POSTITION[1]+noise[1],self.RAG_POSTITION[2], method='direct')
        self.hover_above(self.RAG_POSTITION[0],self.RAG_POSTITION[1],self.RAG_POSTITION[2])

    def clean_paint_brush(self):
        self.dip_brush_in_water()
        self.rub_brush_on_rag()

    def get_paint(self, paint_index):
        x_offset = self.PAINT_DIFFERENCE * np.floor(paint_index/6)
        y_offset = self.PAINT_DIFFERENCE * (paint_index%6)

        x = self.PALLETTE_POSITION[0] + x_offset
        y = self.PALLETTE_POSITION[1] + y_offset
        z = self.PALLETTE_POSITION[2] 

        self.hover_above(x,y,z)
        self.move_to(x,y,z + 0.02, speed=0.2)
        for i in range(3):
            noise = np.clip(np.random.randn(2)*0.0025, a_min=-.005, a_max=0.005)
            self.move_to(x+noise[0],y+noise[1],z, method='direct')
            rate = rospy.Rate(100)
            rate.sleep()
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

    def set_height(self, x, y, z, move_amount=0.0015):
        '''
        Let the user use keyboard keys to lower the paint brush to find 
        how tall something is (z).
        User preses escape to end, then this returns the x, y, z of the end effector
        '''
        import intera_external_devices

        curr_z = z
        curr_x = x
        curr_y = y 

        self.hover_above(curr_x, curr_y, curr_z)
        self.move_to(curr_x, curr_y, curr_z, method='direct')

        print("Controlling height of brush.")
        print("Use w/s for up/down to set the brush to touching the table")
        print("Esc to quit.")

        while not rospy.is_shutdown():
            c = intera_external_devices.getch()
            if c:
                #catch Esc or ctrl-c
                if c in ['\x1b', '\x03']:
                    return curr_x, curr_y, curr_z
                else:
                    if c=='w':
                        curr_z += move_amount
                    elif c=='s':
                        curr_z -= move_amount
                    elif c=='d':
                        curr_x += move_amount
                    elif c=='a':
                        curr_x -= move_amount
                    elif c=='r':
                        curr_y += move_amount
                    elif c=='f':
                        curr_y -= move_amount
                    else:
                        print('Use arrow keys up and down. Esc when done.')
                    
                    self.move_to(curr_x, curr_y,curr_z, method='direct')

    # def set_table_height(self, move_amount=0.0015):
    #     '''
    #     Let the user use the arrow keys to lower the paint brush to find 
    #     how tall the table is (z)
    #     '''
    #     import intera_external_devices
    #     done = False

    #     global INIT_TABLE_Z, TABLE_Z
    #     curr_z = INIT_TABLE_Z
    #     p = canvas_to_global_coordinates(.5, .5, curr_z)
    #     self.hover_above(p[0],p[1],curr_z)
    #     self.move_to(p[0],p[1],curr_z, method='direct')
    #     x, y = .5, .5

    #     print("Controlling height of brush.")
    #     print("Use w/s for up/down to set the brush to touching the table")
    #     print("Esc to quit.")
    #     global q
    #     while not done and not rospy.is_shutdown():
    #         c = intera_external_devices.getch()
    #         if c:
    #             #print('c', c, str(c))
    #             #catch Esc or ctrl-c
    #             if c in ['\x1b', '\x03']:
    #                 #done = True
    #                 print('DONE')
    #                 TABLE_Z = curr_z
    #                 return
    #             else:
    #                 if c=='w':
    #                     # print(curr_z)
    #                     curr_z += move_amount
    #                     # print(curr_z)
    #                     # print("up")
    #                 elif c=='s':
    #                     # print(curr_z)
    #                     curr_z -= move_amount
    #                     # print("down")
    #                 elif c=='d':
    #                     x += move_amount
    #                 elif c=='a':
    #                     x -= move_amount
    #                 elif c=='r':
    #                     y += move_amount
    #                 elif c=='f':
    #                     y -= move_amount
    #                 elif c=='u':
    #                     q[0] += move_amount*2
    #                 elif c=='i':
    #                     q[1] += move_amount*2
    #                 elif c=='o':
    #                     q[2] += move_amount*2
    #                 elif c=='p':
    #                     q[3] += move_amount*2
    #                 elif c=='j':
    #                     q[0] -= move_amount*2
    #                 elif c=='k':
    #                     q[1] -= move_amount*2
    #                 elif c=='l':
    #                     q[2] -= move_amount*2
    #                 elif c==';':
    #                     q[3] -= move_amount*2
    #                 else:
    #                     print('Use arrow keys up and down. Esc when done.')
                    
    #                 p = canvas_to_global_coordinates(x, y, curr_z)
    #                 self.move_to(p[0],p[1],curr_z, method='direct')

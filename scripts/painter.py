#! /usr/bin/env python

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

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
from paint_planner import pick_next_stroke
from strokes import all_strokes
from dslr import WebCam, SimulatedWebCam

try: import rospy
except: pass

q = np.array([0.704020578925, 0.710172716916,0.00244101361829,0.00194372088834])
# q = np.array([0.1,0.2,0.3])
# q = np.array([.9,.155,.127,.05])



from stroke_calibration import process_stroke_library



class Painter():
    '''
        Class that abstracts robot functionality for painting
    '''

    def __init__(self, opt, robot="sawyer", use_cache=False, writer=None):
        '''
        args:
            opt (Options) : from options.py; class with info about painting environment and globals
        kwargs:
            robot ("sawyer" | None) : which robot you are using or None for pure simulation
            use_cache (bool) : Whether to use cached calibration parameters
            writer (TensorBoard) : see tensorboard.py
        '''
        self.opt = opt # Options object

        if not opt.simulate:
            import rospy

        self.robot = None
        if robot == "sawyer":
            self.robot = Sawyer(debug=True)
        elif robot == None:
            self.robot = SimulatedRobot(debug=True)

        self.writer = writer 

        self.robot.good_morning_robot()

        self.curr_position = None
        self.seed_position = None
        self.H_coord = None # Translate coordinates based on faulty camera location

        # self.to_neutral()

        # Set how high the table is wrt the brush
        if use_cache:
            params = pickle.load(open(os.path.join(self.opt.cache_dir, "cached_params.pkl"),'r'))
            self.Z_CANVAS = params['Z_CANVAS']
            self.Z_MAX_CANVAS = params['Z_MAX_CANVAS']
        else:
            print('Brush should be at bottom left of canvas.')
            print('Use keys "w" and "s" to set the brush to just barely touch the canvas.')
            p = canvas_to_global_coordinates(0.5, 0.5, self.opt.INIT_TABLE_Z, self.opt)
            self.Z_CANVAS = self.set_height(p[0], p[1], self.opt.INIT_TABLE_Z)[2]

            # print('Moving brush tip to the top right of canvas.')
            # p = canvas_to_global_coordinates(0.5, 0.5, self.opt.INIT_TABLE_Z, self.opt)
            # self.hover_above(p[0], p[1], self.Z_CANVAS, method='direct')

            print('Move the brush to the lowest point it should go.')
            self.Z_MAX_CANVAS = self.set_height(p[0], p[1], self.Z_CANVAS)[2]
            self.hover_above(p[0], p[1], self.Z_CANVAS, method='direct')

            params = {'Z_CANVAS':self.Z_CANVAS, 'Z_MAX_CANVAS':self.Z_MAX_CANVAS}
            with open(os.path.join(self.opt.cache_dir, 'cached_params.pkl'),'wb') as f:
                pickle.dump(params, f)
            self.to_neutral()
        
        self.Z_RANGE = np.abs(self.Z_MAX_CANVAS - self.Z_CANVAS)

        self.WATER_POSITION = (-.4,.6,self.Z_CANVAS)
        self.RAG_POSTITION = (-.4,.45,self.Z_CANVAS)

        self.PALLETTE_POSITION = (-.3,.5,self.Z_CANVAS- 0.5*self.Z_RANGE)
        self.PAINT_DIFFERENCE = 0.03976

        # Setup Camera
        if not self.opt.simulate:
            self.camera = WebCam(opt)
        else:
            self.camera = SimulatedWebCam(opt)

        if self.camera is not None:
            self.camera.debug = True
            self.camera.calibrate_canvas(use_cache=use_cache)

        img = self.camera.get_canvas()
        self.opt.CANVAS_WIDTH_PIX, self.opt.CANVAS_HEIGHT_PIX = img.shape[1], img.shape[0]

        # Ensure that x,y on the canvas photograph is x,y for the robot interacting with the canvas
        self.coordinate_calibration(use_cache=opt.use_cache)

        # Get brush strokes from stroke library
        if not os.path.exists(os.path.join(self.opt.cache_dir, 'strokes.pkl')) or not use_cache:
            try:
                input('Need to create stroke library. Press enter to start.')
            except SyntaxError:
                pass

            self.paint_stroke_library()
            self.to_neutral()
            self.strokes = process_stroke_library(self.camera.get_canvas(), self.opt)
            with open(os.path.join(self.opt.cache_dir, 'strokes.pkl'),'wb') as f:
                pickle.dump(self.strokes, f)
        else:
            self.strokes = pickle.load(open(os.path.join(self.opt.cache_dir, "strokes.pkl"),'rb'))


        # export the processed strokes for the python3 code
        from export_strokes import export_strokes
        export_strokes(self.opt)

    def next_stroke(self, canvas, target, colors, all_colors, x_y_attempts=5):
        ''' Predict the next brush stroke '''
        return pick_next_stroke(canvas, target, self.strokes, colors, all_colors,
                    H_coord=self.H_coord,
                    x_y_attempts=x_y_attempts)


    def to_neutral(self):
        # Initial spot
        self._move(0.2,0.5,self.opt.INIT_TABLE_Z+0.05, timeout=20, method="direct", speed=0.4)

    def _move(self, x, y, z, timeout=20, method='linear', step_size=.2, speed=0.1):
        if self.opt.simulate: return
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
        self._move(x,y,z+self.opt.HOVER_FACTOR, method=method, speed=0.4)
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
        if self.opt.simulate: return
        self.dip_brush_in_water()
        self.rub_brush_on_rag()

    def get_paint(self, paint_index):
        if self.opt.simulate: return
        x_offset = self.PAINT_DIFFERENCE * np.floor(paint_index/6)
        y_offset = self.PAINT_DIFFERENCE * (paint_index%6)

        x = self.PALLETTE_POSITION[0] + x_offset
        y = self.PALLETTE_POSITION[1] + y_offset
        z = self.PALLETTE_POSITION[2] 

        self.hover_above(x,y,z)
        self.move_to(x,y,z + 0.02, speed=0.2)
        for i in range(3):
            noise = np.clip(np.random.randn(2)*0.0025, a_min=-.008, a_max=0.008)
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

        p0 = canvas_to_global_coordinates(path[0,0], path[0,1], TABLE_Z, self.opt)
        self.hover_above(p0[0], p0[1], TABLE_Z)
        self.move_to(p0[0], p0[1], TABLE_Z + 0.02, speed=0.2)
        p3 = None

        for i in range(1, len(path)-1, 3):
            p1 = canvas_to_global_coordinates(path[i+0,0], path[i+0,1], TABLE_Z, self.opt)
            p2 = canvas_to_global_coordinates(path[i+1,0], path[i+1,1], TABLE_Z, self.opt)
            p3 = canvas_to_global_coordinates(path[i+2,0], path[i+2,1], TABLE_Z, self.opt)

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

        pn = canvas_to_global_coordinates(path[-1,0], path[-1,1], TABLE_Z, self.opt)
        self.move_to(pn[0], pn[1], TABLE_Z + 0.02, speed=0.2)
        self.hover_above(pn[0], pn[1], TABLE_Z)


    def paint_quadratic_bezier(self, p0,p1,p2, step_size=.005):
        p0 = canvas_to_global_coordinates(p0[0], p0[1], TABLE_Z, self.opt)
        p1 = canvas_to_global_coordinates(p1[0], p1[1], TABLE_Z, self.opt)
        p2 = canvas_to_global_coordinates(p2[0], p2[1], TABLE_Z, self.opt)

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

        p = canvas_to_global_coordinates(.5, .5, TABLE_Z, self.opt)
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
        import rospy

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


    def coordinate_calibration(self, debug=True, use_cache=False):
        import matplotlib.pyplot as plt
        from simulated_painting_environment import apply_stroke
        import cv2
        # If you run painter.paint on a given x,y it will be slightly off (both in real and simulation)
        # Close this gap by transforming the given x,y into a coordinate that painter.paint will use
        # to perfectly hit that given x,y using a homograph transformation

        if use_cache and os.path.exists(os.path.join(self.opt.cache_dir, "cached_H_coord.pkl")):
            self.H_coord = pickle.load(open(os.path.join(self.opt.cache_dir, "cached_H_coord.pkl"),'rb'))
            return self.H_coord

        try:
            input('About to calibrate x,y. Please put a fresh piece of paper down, provide black paint (index 0), and press enter when ready.')
        except SyntaxError:
            pass

        # Paint 4 points and compare the x,y's in real vs. sim
        # Compute Homography to compare these

        stroke_ind = 0 # This stroke is pretty much a dot

        canvas = self.camera.get_canvas()
        canvas_width_pix, canvas_height_pix = canvas.shape[1], canvas.shape[0]

        # Points for computing the homography
        t = 0.06 # How far away from corners to paint
        homography_points = [[t,t],[1-t,t],[t,1-t],[1-t,1-t]]


        self.get_paint(0)
        for canvas_coord in homography_points:
            x_prop, y_prop = canvas_coord # Coord in canvas proportions
            x_pix, y_pix = int(x_prop * canvas_width_pix), int((1-y_prop) * canvas_height_pix) #  Coord in canvas pixels
            x_glob,y_glob,_ = canvas_to_global_coordinates(x_prop,y_prop,None, self.opt) # Coord in meters from robot

            # Paint the point
            all_strokes[stroke_ind]().paint(self, x_glob, y_glob, 0)

        # Picture of the new strokes
        self.to_neutral()
        canvas = self.camera.get_canvas()

        sim_coords = []
        real_coords = []
        sim_coords_global = []
        real_coords_global = []
        for canvas_coord in homography_points:
            x_prop, y_prop = canvas_coord 
            x_pix, y_pix = int(x_prop * canvas_width_pix), int((1-y_prop) * canvas_height_pix)

            # Look in the region of the stroke and find the center of the stroke
            w = int(.06 * canvas_height_pix)
            window = canvas[y_pix-w:y_pix+w, x_pix-w:x_pix+w,:]
            window = window.mean(axis=2)
            window /= 255.
            window = 1 - window
            # plt.imshow(window, cmap='gray')
            # plt.show()
            window = window > 0.5
            dark_y, dark_x = window.nonzero()
            x_pix_real = int(np.mean(dark_x)) + x_pix-w
            y_pix_real = int(np.mean(dark_y)) + y_pix-w

            real_coords.append(np.array([x_pix_real, y_pix_real]))
            sim_coords.append(np.array([x_pix, y_pix]))

            # Coord in meters from robot
            x_sim_glob,y_sim_glob,_ = canvas_to_global_coordinates(x_prop,y_prop,None, self.opt) 
            sim_coords_global.append(np.array([x_sim_glob, y_sim_glob]))
            x_real_glob,y_real_glob,_ = canvas_to_global_coordinates(1.*x_pix_real/canvas_width_pix,\
                    1-(1.*y_pix_real/canvas_height_pix),None, self.opt) 
            real_coords_global.append(np.array([x_real_glob, y_real_glob]))

        real_coords, sim_coords = np.array(real_coords), np.array(sim_coords)
        real_coords_global, sim_coords_global = np.array(real_coords_global), np.array(sim_coords_global)
        
        # H, _ = cv2.findHomography(real_coords, sim_coords)      
        H, _ = cv2.findHomography(real_coords_global, sim_coords_global)  
        # canvas_warp = cv2.warpPerspective(canvas.copy(), H, (canvas.shape[1], canvas.shape[0]))

        # if debug:
        #     fix, ax = plt.subplots(1,2)
        #     ax[0].imshow(canvas)
        #     ax[0].scatter(real_coords[:,0], real_coords[:,1], c='r')
        #     ax[0].scatter(sim_coords[:,0], sim_coords[:,1], c='g')
        #     ax[0].set_title('non-transformed photo')

        #     ax[1].imshow(canvas_warp)
        #     ax[1].scatter(real_coords[:,0], real_coords[:,1], c='r')
        #     ax[1].scatter(sim_coords[:,0], sim_coords[:,1], c='g')
        #     ax[1].set_title('warped photo')
        #     plt.show()
        # if debug:
        #     plt.imshow(canvas)
        #     plt.scatter(real_coords[:,0], real_coords[:,1], c='r')
        #     plt.scatter(sim_coords[:,0], sim_coords[:,1], c='g')
        #     sim_coords = np.array([int(.5*canvas_width_pix),int(.5*canvas_height_pix),1.])
        #     real_coords = H.dot(sim_coords)
        #     real_coords /= real_coords[2]
        #     plt.scatter(real_coords[0], real_coords[1], c='r')
        #     plt.scatter(sim_coords[0], sim_coords[1], c='g')
        #     plt.show()

        self.H_coord = H
        # Cache it
        with open(os.path.join(self.opt.cache_dir, 'cached_H_coord.pkl'),'wb') as f:
            pickle.dump(self.H_coord, f)
        return H

    def paint_stroke_library(self):
        strokes = all_strokes
        stroke_ind = 0
        
        rotation = 0
        forbidden = ((0,0), (0,self.opt.cells_x-1), (self.opt.cells_y-1,0), (self.opt.cells_y-1, self.opt.cells_x-1))
        for i in range(self.opt.cells_x):
            for j in range(self.opt.cells_y):
                if (j,i) in forbidden: continue
                if stroke_ind >= len(strokes): 
                    # stroke_ind=0
                    break
                    # rotation += 3.14*.25
                if stroke_ind % 4 == 0:
                    self.clean_paint_brush()
                    self.get_paint(0)

                # Get the position of the start of the stroke
                x = self.opt.CANVAS_POSITION[0] - 0.5*self.opt.CANVAS_WIDTH + i * self.opt.cell_dim_x + self.opt.over
                y = self.opt.CANVAS_POSITION[1] + self.opt.CANVAS_HEIGHT - j*self.opt.cell_dim_y - self.opt.down
                
                stroke = strokes[stroke_ind]()
                #print(stroke)
                stroke.paint(self, x, y, rotation)

                stroke_ind += 1
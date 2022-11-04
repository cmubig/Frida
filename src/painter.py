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
# import scipy.special
import pickle
import math
import gzip
import subprocess

from paint_utils import *
from robot import *
from painting_materials import *
from strokes import all_strokes, get_base_strokes, get_random_stroke, simple_parameterization_to_real
from camera.dslr import WebCam, SimulatedWebCam

try: import rospy
except: pass

q = np.array([0.704020578925, 0.710172716916,0.00244101361829,0.00194372088834])
# q = np.array([0.1,0.2,0.3])
# q = np.array([.9,.155,.127,.05])


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

        # Save the first neutral join positions. This will let us get back to 
        # a clean state for neutral each time
        self.og_neutral = None

        if not opt.simulate:
            import rospy

        self.robot = None
        if robot == "sawyer":
            self.robot = Sawyer(debug=True)
        elif robot == None:
            self.robot = SimulatedRobot(debug=True)

        self.writer = writer 

        self.robot.good_morning_robot()


        # Setup Camera
        while True:
            try:
                if not self.opt.simulate:
                    self.camera = WebCam(opt)
                else:
                    self.camera = SimulatedWebCam(opt)
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(e)
                try:
                    input('Could not connect camera. Try turning it off and on, then press start.')
                except SyntaxError:
                    pass

        self.curr_position = None
        self.seed_position = None
        self.H_coord = None # Translate coordinates based on faulty camera location

        p = canvas_to_global_coordinates(0, 0.5, self.opt.INIT_TABLE_Z, self.opt)
        self.move_to(p[0], p[1], self.opt.INIT_TABLE_Z, speed=0.2)
        self.to_neutral()

        # Set how high the table is wrt the brush
        if use_cache and os.path.exists(os.path.join(self.opt.cache_dir, "cached_params.pkl")):
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

        self.WATER_POSITION = (-.4,.58,self.Z_CANVAS)
        self.RAG_POSTITION = (-.42,.41,self.Z_CANVAS)

        self.PALLETTE_POSITION = (-.3,.47,self.Z_CANVAS- 0.5*self.Z_RANGE)
        self.PAINT_DIFFERENCE = 0.03976


        # self.locate_canvas()
        # self.calibrate_robot_tilt()


        if self.camera is not None:
            self.camera.debug = True
            self.camera.calibrate_canvas(use_cache=use_cache)

        img = self.camera.get_canvas()
        self.opt.CANVAS_WIDTH_PIX, self.opt.CANVAS_HEIGHT_PIX = img.shape[1], img.shape[0]

        # Ensure that x,y on the canvas photograph is x,y for the robot interacting with the canvas
        self.coordinate_calibration(use_cache=opt.use_cache)

        # Get brush strokes from stroke library
        if not os.path.exists(os.path.join(self.opt.cache_dir, 'extended_stroke_library_intensities.npy')) or not use_cache:
            if not opt.simulate:
                try:
                    input('Need to create stroke library. Press enter to start.')
                except SyntaxError:
                    pass
                self.paint_extended_stroke_library()
        if not os.path.exists(os.path.join(self.opt.cache_dir, 'param2img.pt')) or not use_cache:
            if not self.opt.dont_retrain_stroke_model:
                self.create_continuous_stroke_model()



    def to_neutral(self, speed=0.4):
        # Initial spot
        x, y, z = -0.18, 0.5, self.opt.INIT_TABLE_Z+0.05
        if self.og_neutral is None:
            self.og_neutral = self._move(x, y, z, timeout=20, method="direct", speed=speed)
        else:
            self.robot.move_to_joint_positions(self.og_neutral, timeout=20, speed=speed)
            self.seed_position = self.og_neutral
        self.curr_position = [x, y, z]

    def _move(self, x, y, z, timeout=20, method='direct', step_size=.1, speed=0.1):
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

        method = 'linear' ##############################################################################
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
                    self.seed_position = None
        elif method == 'curved':
            # TODO
            pass
        else:
            # Direct
            pos = self.robot.inverse_kinematics([x, y, z], q, seed_position=self.seed_position)
            self.seed_position = pos
            try:
                self.robot.move_to_joint_positions(pos, timeout=timeout, speed=speed)
            except Exception as e:
                print("error moving robot: ", e)
                self.seed_position = None

        self.curr_position = [x, y, z]
        return pos

    def hover_above(self, x,y,z, method='direct'):
        self._move(x,y,z+self.opt.HOVER_FACTOR, method=method, speed=0.4)
        # rate = rospy.Rate(100)
        # rate.sleep()

    def move_to(self, x,y,z, method='direct', speed=0.05):
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
            noise = np.clip(np.random.randn(2)*0.04, a_min=-.04, a_max=0.04)
            self.move_to(self.RAG_POSTITION[0]+noise[0],self.RAG_POSTITION[1]+noise[1],self.RAG_POSTITION[2], method='linear')
        self.hover_above(self.RAG_POSTITION[0],self.RAG_POSTITION[1],self.RAG_POSTITION[2])

    def clean_paint_brush(self):
        if self.opt.simulate: return
        self.move_to(self.WATER_POSITION[0],self.WATER_POSITION[1],self.WATER_POSITION[2]+0.09, speed=0.3)
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

    def calibrate_robot_tilt(self):

        while(True):
            self._move(-.6,.5,self.Z_CANVAS+.09,  speed=0.2)
            self._move(-.6,.5,self.Z_CANVAS,  speed=0.2)
            try:
                input('press enter to move to next position')
            except SyntaxError:
                pass
            self._move(-.6,.5,self.Z_CANVAS+.09,  speed=0.2)
            self._move(.5,.5,self.Z_CANVAS+.09,  speed=0.2)
            self._move(.5,.5,self.Z_CANVAS,  speed=0.2)
            try:
                input('press enter to move to next position')
            except SyntaxError:
                pass

            self._move(.5,.5,self.Z_CANVAS+.09,  speed=0.2)
            self.to_neutral(speed=0.2)
            self._move(0,.35,self.Z_CANVAS+.09,  speed=0.2)
            self._move(0,.35,self.Z_CANVAS,  speed=0.2)
            try:
                input('press enter to move to next position')
            except SyntaxError:
                pass
            self._move(0,.35,self.Z_CANVAS+.09,  speed=0.2)
            self.to_neutral(speed=0.2)
            self._move(0,.8,self.Z_CANVAS+.09,  speed=0.2)
            self._move(0,.8,self.Z_CANVAS,  speed=0.2)
            try:
                input('press enter to move to next position')
            except SyntaxError:
                pass
            self._move(0,.8,self.Z_CANVAS+.09,  speed=0.2)

    def locate_canvas(self):

        while(True):

            for x in [0.0, 1.0]:
                for y in [0.0, 1.0]:
                    x_glob,y_glob,_ = canvas_to_global_coordinates(x,y,None, self.opt) # Coord in meters from robot
                    self._move(x_glob,y_glob,self.Z_CANVAS+.02,  speed=0.1)
                    self._move(x_glob,y_glob,self.Z_CANVAS+.005,  speed=0.05)
                    self._move(x_glob,y_glob,self.Z_CANVAS+.02,  speed=0.1)


    def coordinate_calibration(self, debug=True, use_cache=False):
        import matplotlib.pyplot as plt
        from scipy.ndimage import median_filter
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
        canvas_og = canvas.copy()
        canvas_width_pix, canvas_height_pix = canvas.shape[1], canvas.shape[0]

        # Points for computing the homography
        t = 0.1 # How far away from corners to paint
        # homography_points = [[t,t],[1-t,t],[t,1-t],[1-t,1-t]]
        homography_points = []
        for i in np.linspace(0.1, 0.9, 4):
            for j in np.linspace(0.1, 0.9, 4):
                homography_points.append([i,j])

        
        i = 0
        for canvas_coord in homography_points:
            if i % 5 == 0: self.get_paint(0)
            i += 1
            x_prop, y_prop = canvas_coord # Coord in canvas proportions
            # x_pix, y_pix = int(x_prop * canvas_width_pix), int((1-y_prop) * canvas_height_pix) #  Coord in canvas pixels
            x_glob,y_glob,_ = canvas_to_global_coordinates(x_prop,y_prop,None, self.opt) # Coord in meters from robot

            # Paint the point
            all_strokes[stroke_ind]().paint(self, x_glob, y_glob, 0)

        # Picture of the new strokes
        self.to_neutral()
        canvas = self.camera.get_canvas().astype(np.float32)

        sim_coords = []
        real_coords = []
        sim_coords_global = []
        real_coords_global = []
        for canvas_coord in homography_points:
            try:
                x_prop, y_prop = canvas_coord 
                x_pix, y_pix = int(x_prop * canvas_width_pix), int((1-y_prop) * canvas_height_pix)

                # Look in the region of the stroke and find the center of the stroke
                w = int(.1 * canvas_height_pix)
                window = canvas[y_pix-w:y_pix+w, x_pix-w:x_pix+w,:]
                window = window.mean(axis=2)
                window /= 255.
                window = 1 - window
                # plt.imshow(window, cmap='gray', vmin=0, vmax=1)
                # plt.show()
                window = window > 0.5
                window[:int(0.05*window.shape[0])] = 0
                window[int(0.95*window.shape[0]):] = 0
                window[:,:int(0.05*window.shape[1])] = 0
                window[:,int(0.95*window.shape[1]):] = 0
                window = median_filter(window, size=(9,9))
                dark_y, dark_x = window.nonzero()
                # plt.matshow(window)
                # plt.scatter(int(np.median(dark_x)), int(np.median(dark_y)))
                # plt.show()
                # fig, ax = plt.subplots(1)
                fig = plt.figure()
                ax = fig.gca()
                ax.matshow(window)
                ax.scatter(int(np.median(dark_x)), int(np.median(dark_y)))
                self.writer.add_figure('coordinate_homography/{}'.format(len(real_coords_global)), fig, 0)

                x_pix_real = int(np.median(dark_x)) + x_pix-w
                y_pix_real = int(np.median(dark_y)) + y_pix-w

                real_coords.append(np.array([x_pix_real, y_pix_real]))
                sim_coords.append(np.array([x_pix, y_pix]))

                # Coord in meters from robot
                x_sim_glob,y_sim_glob,_ = canvas_to_global_coordinates(x_prop,y_prop,None, self.opt) 
                sim_coords_global.append(np.array([x_sim_glob, y_sim_glob]))
                x_real_glob,y_real_glob,_ = canvas_to_global_coordinates(1.*x_pix_real/canvas_width_pix,\
                        1-(1.*y_pix_real/canvas_height_pix),None, self.opt) 
                real_coords_global.append(np.array([x_real_glob, y_real_glob]))
            except Exception as e:
                print(e)
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
                if stroke_ind % 6 == 0:
                    self.clean_paint_brush()
                if stroke_ind % 3 == 0:
                    self.get_paint(0)

                # Get the position of the start of the stroke
                x = self.opt.CANVAS_POSITION[0] - 0.5*self.opt.CANVAS_WIDTH + i * self.opt.cell_dim_x + self.opt.over
                y = self.opt.CANVAS_POSITION[1] + self.opt.CANVAS_HEIGHT - j*self.opt.cell_dim_y - self.opt.down
                
                stroke = strokes[stroke_ind]()
                #print(stroke)
                stroke.paint(self, x, y, rotation)

                stroke_ind += 1


    def paint_extended_stroke_library(self, max_stroke_meters=0.05):
        w = self.opt.CANVAS_WIDTH_PIX
        h = self.opt.CANVAS_HEIGHT_PIX

        def shift_image(X, dx, dy):
            X = np.roll(X, dy, axis=0)
            X = np.roll(X, dx, axis=1)
            if dy>0:
                X[:dy, :] = 255
            elif dy<0:
                X[dy:, :] = 255
            if dx>0:
                X[:, :dx] = 255
            elif dx<0:
                X[:, dx:] = 255
            return X

        self.to_neutral()
        canvas_without_stroke = self.camera.get_canvas()
        strokes_without_getting_new_paint = 999 
        strokes_without_cleaning = 9999

        # Some pre-programmed strokes to start
        random_strokes = []
        random_strokes.append(simple_parameterization_to_real(.04, .02, 0.5))
        r = 5
        lengths = np.arange(r, dtype=np.float32)/(r-1)*(0.05-0.01) + 0.01
        bends = np.arange(r, dtype=np.float32)/(r-1)*0.04 - 0.02
        zs = np.arange(r, dtype=np.float32)/(r-1)
        for i in range(r):
            random_strokes.append(simple_parameterization_to_real(lengths[i], .02, 0.5))
        for i in range(r):
            random_strokes.append(simple_parameterization_to_real(.04, bends[i], 0.5))
        for i in range(r):
            random_strokes.append(simple_parameterization_to_real(.04, .02, zs[i]))

        # Figure out how many strokes can be made on the given canvas size
        n_strokes_x = int(math.floor(self.opt.CANVAS_WIDTH/(max_stroke_meters)))
        n_strokes_y = int(math.floor(self.opt.CANVAS_HEIGHT/0.03))

        stroke_trajectories = [] # Flatten so it's just twelve values x0,y0,z0,x1,y1,...
        stroke_intensities = [] # list of 2D numpy arrays 0-1 where 1 is paint
        for paper_it in range(self.opt.num_papers):
            for y_offset_pix_og in np.linspace((.03/self.opt.CANVAS_HEIGHT), 0.99-(.02/self.opt.CANVAS_HEIGHT), n_strokes_y)*h:
                for x_offset_pix in np.linspace(0.02, 0.99-(max_stroke_meters/self.opt.CANVAS_WIDTH), n_strokes_x)*w:

                    if len(stroke_trajectories) % 2 == 0:
                        y_offset_pix = y_offset_pix_og + 0.01 * (h/self.opt.CANVAS_HEIGHT) # Offset y by 1cm every other stroke
                    else:
                        y_offset_pix = y_offset_pix_og

                    x, y = x_offset_pix / w, 1 - (y_offset_pix / h)
                    x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
                    x,y,_ = canvas_to_global_coordinates(x,y,None,self.opt)

                    
                    if strokes_without_cleaning >= 8:
                        self.clean_paint_brush()
                        self.get_paint(0)
                        strokes_without_cleaning, strokes_without_getting_new_paint = 0, 0
                    if strokes_without_getting_new_paint >= 4:
                        self.get_paint(0)
                        strokes_without_getting_new_paint = 0
                    strokes_without_getting_new_paint += 1
                    strokes_without_cleaning += 1

                    if len(stroke_trajectories) < len(random_strokes):
                        random_stroke = random_strokes[len(stroke_trajectories)]
                    else:
                        random_stroke = get_random_stroke()
                    random_stroke.paint(self, x, y, 0)

                    self.to_neutral()
                    canvas_with_stroke = self.camera.get_canvas()

                    stroke = canvas_with_stroke.copy()
                    # show_img(stroke)

                    #diff = np.mean(np.abs(canvas_with_stroke - canvas_without_stroke), axis=2)
                    og_shape = canvas_without_stroke.shape
                    canvas_before_ = cv2.resize(canvas_without_stroke, (256,256)) # For scipy memory error
                    canvas_after_ = cv2.resize(canvas_with_stroke, (256,256)) # For scipy memory error

                    diff = np.max(np.abs(canvas_after_.astype(np.float32) - canvas_before_.astype(np.float32)), axis=2)
                    # diff = diff / 255.#diff.max()

                    # smooth the diff
                    diff = median_filter(diff, size=(3,3))
                    diff = cv2.resize(diff,  (og_shape[1], og_shape[0]))

                    stroke[diff < 20, :] = 255.
                    # show_img(stroke)

                    # Translate canvas so that the current stroke is directly in the middle of the canvas
                    stroke = shift_image(stroke.copy(), 
                        dx=int(.5*w - x_offset_pix), 
                        dy=int(.5*h - y_offset_pix))
                    
                    stroke[stroke > 190] = 255
                    # show_img(stroke)
                    stroke = stroke / 255.
                    # show_img(stroke)
                    stroke = 1 - stroke 
                    # show_img(stroke)
                    stroke = stroke.mean(axis=2)
                    # show_img(stroke)
                    if stroke.max() < 0.1 or stroke.min() > 0.8:
                        continue # Didn't make a mark
                    # stroke /= stroke.max()
                    # show_img(stroke)

                    if len(stroke_intensities) % 5 == 0:
                        fig = plt.figure()
                        ax = fig.gca()
                        ax.scatter(stroke.shape[1]/2, stroke.shape[0]/2)
                        ax.imshow(stroke, cmap='gray', vmin=0, vmax=1)
                        ax.set_xticks([]), ax.set_yticks([])
                        fig.tight_layout()
                        self.writer.add_figure('stroke_library/{}'.format(len(stroke_intensities)), fig, 0)

                    stroke[:int(.3*h)] = 0 
                    stroke[int(.6*h)] = 0
                    stroke[:,:int(.4*w)] = 0 
                    stroke[:,int(.8*w):] = 0


                    # plt.scatter(int(w*.5), int(h*.5))
                    # show_img(stroke)

                    traj = np.array(random_stroke.trajectory).flatten()
                    # print(traj, '\n', random_stroke.trajectory)
                    stroke_intensities.append(stroke)
                    stroke_trajectories.append(traj)

                    canvas_without_stroke = canvas_with_stroke.copy()

                    if len(stroke_intensities) % 5 == 0:
                        # Save data
                        with open(os.path.join(self.opt.cache_dir, 'extended_stroke_library_trajectories.npy'), 'wb') as f:
                            np.save(f, np.stack(stroke_trajectories, axis=0))
                        # with open(os.path.join(self.opt.cache_dir, 'extended_stroke_library_intensities.npy'), 'wb') as f:
                        #     intensities = np.stack(np.stack(stroke_intensities, axis=0), axis=0)
                        #     intensities = (intensities * 255).astype(np.uint8)
                        #     np.save(f, intensities)
                        with gzip.GzipFile(os.path.join(self.opt.cache_dir, 'extended_stroke_library_intensities.npy'), 'w') as f:
                            intensities = np.stack(np.stack(stroke_intensities, axis=0), axis=0)
                            intensities = (intensities * 255).astype(np.uint8)
                            np.save(f, intensities)
                        with open(os.path.join(self.opt.cache_dir, 'stroke_size.npy'), 'wb') as f:
                            np.save(f, np.array(stroke_intensities[0].shape))

            if paper_it != self.opt.num_papers-1:
                try:
                    input('Place down new paper. Press enter to start.')
                except SyntaxError:
                    pass

    def create_continuous_stroke_model(self):
        # Call a script to take the stroke library and model it using a neural network
        # It will save the model to a file to be used later
        # must be run in python3
        import rospkg
        rospack = rospkg.RosPack()
        # get the file path for painter code
        ros_dir = rospack.get_path('paint')

        exit_code = subprocess.call(['python3', 
            os.path.join(ros_dir, 'src','continuous_brush_model.py')]\
            +sys.argv[1:])

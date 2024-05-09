#! /usr/bin/env python

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import os
import shutil
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.ndimage import median_filter
import pickle
import math
import gzip
from brush_stroke import BrushStroke

from paint_utils3 import canvas_to_global_coordinates
from robot import *
from camera.dslr import WebCam, SimulatedWebCam


PERPENDICULAR_QUATERNION = np.array([1.77622069e-04,   9.23865441e-01,  -3.82717408e-01,  -1.73978366e-05])

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

class Painter():
    '''
        Class that abstracts robot functionality for painting
    '''

    def __init__(self, opt):
        '''
        args:
            opt (Options) : from options.py; class with info about painting environment and globals
        kwargs:
        '''
        self.opt = opt # Options object
        use_cache = opt.use_cache

        self.robot = None
        if opt.simulate:
            self.robot = SimulatedRobot(debug=True)
        elif opt.robot == "franka":
            self.robot = Franka(debug=True)
        elif opt.robot == "xarm":
            self.robot = XArm(opt.xarm_ip, debug=True)
        elif opt.robot == None:
            self.robot = SimulatedRobot(debug=True)
        if opt.simulate:
            self.robot = SimulatedRobot(debug=True)

        self.writer = opt.writer # TODO simplify

        self.robot.good_morning_robot()

        # Setup Camera
        while True: 
            try:
                if not self.opt.simulate and not opt.no_camera:
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

        self.H_coord = None # Translate coordinates based on faulty camera location

        self.to_neutral()

        # Set how high the table is wrt the brush
        if use_cache and os.path.exists(os.path.join(self.opt.cache_dir, "brush_tip_to_table_calib.json")):
            params = json.load(open(os.path.join(self.opt.cache_dir, "brush_tip_to_table_calib.json"),'rb'))
            self.Z_CANVAS = params['Z_CANVAS']
            self.Z_MAX_CANVAS = params['Z_MAX_CANVAS']
        else:
            print('\nBrush tip calibration\n')
            print('Brush should be at very center of the canvas.')
            print('Use keys "w" and "s" to set the brush to just bare/localy touch the canvas.')
            p = canvas_to_global_coordinates(0.5, 0.5, self.opt.INIT_TABLE_Z, self.opt)
            self.Z_CANVAS = self.set_height(p[0], p[1], self.opt.INIT_TABLE_Z)[2]

            print('Move the brush to the lowest point it should go again with "w" and "s" keys.')
            self.Z_MAX_CANVAS = self.set_height(p[0], p[1], self.Z_CANVAS)[2]
            self.hover_above(p[0], p[1], self.Z_CANVAS, method='direct')

            params = {'Z_CANVAS':self.Z_CANVAS, 'Z_MAX_CANVAS':self.Z_MAX_CANVAS}
            if not os.path.exists(self.opt.cache_dir):
                os.mkdir(self.opt.cache_dir)
            with open(os.path.join(self.opt.cache_dir, 'brush_tip_to_table_calib.json'),'w') as f:
                json.dump(params, f, indent=4)
            self.to_neutral()

        self.Z_RANGE = np.abs(self.Z_MAX_CANVAS - self.Z_CANVAS)

        self.opt.WATER_POSITION.append(self.Z_CANVAS)
        self.opt.RAG_POSTITION.append(self.Z_CANVAS)

        self.opt.PALLETTE_POSITION.append(self.Z_CANVAS- 0.2*self.Z_RANGE)
        # self.opt.PAINT_DIFFERENCE = 0.03976

        # while True: 
        #     self.locate_items()

        # self.locate_canvas()
        # self.calibrate_robot_tilt()

        # if self.opt.ink:
        #     corners = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]).astype(np.float32)
        #     for i in range(len(corners)):
        #         corner = corners[i]
        #         x, y = corner[0], corner[1]
        #         x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
        #         x,y,_ = canvas_to_global_coordinates(x,y,None,self.opt)
        #         if i == 0:
        #             self.move_to(x,y,self.Z_CANVAS+0.03)
        #         self.move_to(x,y,self.Z_CANVAS)
        #         if i == 4:
        #             self.move_to(x,y,self.Z_CANVAS+0.03)
        #     self.to_neutral()

        if self.camera is not None:
            self.camera.debug = True
            self.camera.calibrate_canvas(use_cache=use_cache)

        img = self.camera.get_canvas()
        self.opt.CANVAS_WIDTH_PIX, self.opt.CANVAS_HEIGHT_PIX = img.shape[1], img.shape[0]

        # Ensure that x,y on the canvas photograph is x,y for the robot interacting with the canvas
        self.coordinate_calibration(use_cache=opt.use_cache)

        # self.paint_fill_in_library() ######################################################

        # Get brush strokes from stroke library
        stroke_lib_exists = os.path.exists(os.path.join(self.opt.cache_dir, 'stroke_library'))
        if stroke_lib_exists:
            stroke_lib_exists = len(os.listdir(os.path.join(self.opt.cache_dir, 'stroke_library')))
        if not stroke_lib_exists or not use_cache:
            if not opt.simulate:
                try:
                    input('Need to create stroke library. Press enter to start.')
                except SyntaxError:
                    pass
                # self.paint_fill_in_library()
                self.paint_extended_stroke_library()
        
        # We must use the same settings as when the stroke library was created. 
        # e.g., we cannot make strokes longer than the ones that we trained on
        with open(os.path.join(self.opt.cache_dir, 'stroke_library', 'stroke_settings_during_library.json'), 'r') as f:
            settings = json.load(f)
            print(settings)
            self.opt.MAX_BEND = settings['MAX_BEND']
            self.opt.MIN_STROKE_Z = settings['MIN_STROKE_Z']
            self.opt.MIN_STROKE_LENGTH = settings['MIN_STROKE_LENGTH']
            self.opt.MAX_STROKE_LENGTH = settings['MAX_STROKE_LENGTH']
            self.opt.MAX_ALPHA = settings['MAX_ALPHA']
            self.opt.STROKE_LIBRARY_CANVAS_WIDTH_M = settings['CANVAS_WIDTH_M']
            self.opt.STROKE_LIBRARY_CANVAS_HEIGHT_M = settings['CANVAS_HEIGHT_M']

        # if not os.path.exists(os.path.join(self.opt.cache_dir, 'param2img.pt')) or not use_cache:
        #     if not self.opt.dont_retrain_stroke_model:
        if not use_cache or not self.opt.dont_retrain_stroke_model:
            from param2stroke import train_param2stroke
            train_param2stroke(self.opt)



    def to_neutral(self, speed=0.4):
        # Initial spot
        if not self.opt.simulate:
            # self.robot.fa.reset_joints()
            y = 0.4 
            if self.opt.robot == 'franka':
                y = 0.4
            elif self.opt.robot == 'xarm':
                y = 0.1
            self.move_to_trajectories([[0,y,self.opt.INIT_TABLE_Z]], [None])

    def move_to_trajectories(self, positions, orientations):
        for i in range(len(orientations)):
            if orientations[i] is None:
                orientations[i] = PERPENDICULAR_QUATERNION
        return self.robot.go_to_cartesian_pose(positions, orientations)

    def _move(self, x, y, z, q=None, timeout=20, method='direct', 
            step_size=.1, speed=0.1, duration=5):
        if self.opt.simulate: return
        '''
        Move to given x, y, z in global coordinates
        kargs:
            method 'linear'|'curved'|'direct'
        '''

        if q is None:
            q = PERPENDICULAR_QUATERNION

        self.robot.go_to_cartesian_pose([x,y,z], q)
        return None

    def hover_above(self, x,y,z, method='direct'):
        self._move(x,y,z+self.opt.HOVER_FACTOR, method=method, speed=0.4)

    def move_to(self, x,y,z, q=None, method='direct', speed=0.05):
        self._move(x,y,z, q=q, method=method, speed=speed)

    def locate_items(self):
        self.dip_brush_in_water()
        self.rub_brush_on_rag()
        self.get_paint(0)
        self.get_paint(5)
        self.get_paint(6)
        self.get_paint(11)
        import time
        p = canvas_to_global_coordinates(0, 0, self.Z_CANVAS, self.opt)
        self.hover_above(p[0], p[1], p[2])
        self.move_to(p[0], p[1], p[2])
        time.sleep(1)
        p = canvas_to_global_coordinates(0, 1, self.Z_CANVAS, self.opt)
        self.move_to(p[0], p[1], p[2])
        time.sleep(1)
        p = canvas_to_global_coordinates(1, 1, self.Z_CANVAS, self.opt)
        self.move_to(p[0], p[1], p[2])
        time.sleep(1)
        p = canvas_to_global_coordinates(1, 0, self.Z_CANVAS, self.opt)
        self.move_to(p[0], p[1], p[2])
        time.sleep(1)
        p = canvas_to_global_coordinates(0, 0, self.Z_CANVAS, self.opt)
        self.move_to(p[0], p[1], p[2])
        time.sleep(1)
        self.hover_above(p[0], p[1], p[2])

    def dip_brush_in_water(self):
        self.move_to(self.opt.RAG_POSTITION[0],self.opt.RAG_POSTITION[1],self.opt.RAG_POSTITION[2]+self.opt.HOVER_FACTOR)
        self.move_to(self.opt.WATER_POSITION[0],self.opt.WATER_POSITION[1],self.opt.WATER_POSITION[2]+self.opt.HOVER_FACTOR)
        positions = []
        positions.append([self.opt.WATER_POSITION[0],self.opt.WATER_POSITION[1],self.opt.WATER_POSITION[2]+self.opt.HOVER_FACTOR])
        positions.append([self.opt.WATER_POSITION[0],self.opt.WATER_POSITION[1],self.opt.WATER_POSITION[2]])
        for i in range(5):
            noise = np.clip(np.random.randn(2)*0.01, a_min=-.02, a_max=0.02)
            positions.append([self.opt.WATER_POSITION[0]+noise[0],self.opt.WATER_POSITION[1]+noise[1],self.opt.WATER_POSITION[2]])
        positions.append([self.opt.WATER_POSITION[0],self.opt.WATER_POSITION[1],self.opt.WATER_POSITION[2]+self.opt.HOVER_FACTOR])
        orientations = [None]*len(positions)
        self.move_to_trajectories(positions, orientations)

    def rub_brush_on_rag(self):
        self.move_to(self.opt.RAG_POSTITION[0],self.opt.RAG_POSTITION[1],self.opt.RAG_POSTITION[2]+self.opt.HOVER_FACTOR, speed=0.3)
        positions = []
        positions.append([self.opt.RAG_POSTITION[0],self.opt.RAG_POSTITION[1],self.opt.RAG_POSTITION[2]+self.opt.HOVER_FACTOR])
        positions.append([self.opt.RAG_POSTITION[0],self.opt.RAG_POSTITION[1],self.opt.RAG_POSTITION[2]])
        for i in range(5):
            noise = np.clip(np.random.randn(2)*0.06, a_min=-.06, a_max=0.06)
            positions.append([self.opt.RAG_POSTITION[0]+noise[0],self.opt.RAG_POSTITION[1]+noise[1],self.opt.RAG_POSTITION[2]])
        positions.append([self.opt.RAG_POSTITION[0],self.opt.RAG_POSTITION[1],self.opt.RAG_POSTITION[2]+self.opt.HOVER_FACTOR])
        orientations = [None]*len(positions)
        self.move_to_trajectories(positions, orientations)
    
    def touch_rag(self):
        self.move_to(self.opt.RAG_POSTITION[0],self.opt.RAG_POSTITION[1],self.opt.RAG_POSTITION[2]+self.opt.HOVER_FACTOR, speed=0.3)
        positions = []
        noise = np.clip(np.random.randn(2)*0.04, a_min=-.04, a_max=0.04)
        positions.append([self.opt.RAG_POSTITION[0]+noise[0],self.opt.RAG_POSTITION[1]+noise[1],self.opt.RAG_POSTITION[2]+self.opt.HOVER_FACTOR])
        positions.append([self.opt.RAG_POSTITION[0]+noise[0],self.opt.RAG_POSTITION[1]+noise[1],self.opt.RAG_POSTITION[2]+0.005])
        positions.append([self.opt.RAG_POSTITION[0],self.opt.RAG_POSTITION[1],self.opt.RAG_POSTITION[2]+self.opt.HOVER_FACTOR])
        orientations = [None]*len(positions)
        self.move_to_trajectories(positions, orientations)

    def clean_paint_brush(self):
        if self.opt.simulate: return
        self.dip_brush_in_water()
        self.rub_brush_on_rag()

    def get_paint(self, paint_index):
        if self.opt.simulate: return
        x_offset = self.opt.PAINT_DIFFERENCE * np.floor(paint_index/6)
        y_offset = self.opt.PAINT_DIFFERENCE * (paint_index%6)

        x = self.opt.PALLETTE_POSITION[0] + x_offset
        y = self.opt.PALLETTE_POSITION[1] + y_offset
        z = self.opt.PALLETTE_POSITION[2] 
        
        self.move_to(x,y,z+self.opt.HOVER_FACTOR)

        positions, orientations = [], []
        positions.append([x,y,z+self.opt.HOVER_FACTOR])
        positions.append([x,y,z+0.04])
        for i in range(3):
            noise = np.clip(np.random.randn(2)*0.004, a_min=-.009, a_max=0.009)
            positions.append([x+noise[0],y+noise[1],z])
        positions.append([x,y,z + 0.02])
        positions.append([x,y,z+self.opt.HOVER_FACTOR])
        orientations = [None]*len(positions)
        self.move_to_trajectories(positions, orientations)
        self.touch_rag()


    def set_height(self, x, y, z, move_amount=0.0015):
        '''
        Let the user use keyboard keys to lower the paint brush to find 
        how tall something is (z).
        User preses escape to end, then this returns the x, y, z of the end effector
        '''
        
        import getch

        curr_z = z
        curr_x = x
        curr_y = y 

        self.hover_above(curr_x, curr_y, curr_z)
        self.move_to(curr_x, curr_y, curr_z, method='direct')

        print("Controlling height of brush.")
        print("Use w/s for up/down to set the brush to touching the table")
        print("Esc to quit.")

        while True:
            c = getch.getch()
            
            if c:
                # print(c)
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
                    
                    self._move(curr_x, curr_y,curr_z)

    def calibrate_robot_tilt(self):

        while(True):
            self._move(-.5,.53,self.Z_CANVAS+.09,  speed=0.2)
            self._move(-.5,.53,self.Z_CANVAS,  speed=0.2)
            try:
                input('press enter to move to next position')
            except SyntaxError:
                pass
            self._move(-.6,.53,self.Z_CANVAS+.09,  speed=0.2)
            self._move(.5,.5,self.Z_CANVAS+.09,  speed=0.2)
            self._move(.5,.5,self.Z_CANVAS,  speed=0.2)
            try:
                input('press enter to move to next position')
            except SyntaxError:
                pass

            self._move(.5,.5,self.Z_CANVAS+.09,  speed=0.2)
            # self.to_neutral(speed=0.2)
            self._move(0,.4,self.Z_CANVAS+.09,  speed=0.2)
            self._move(0,.4,self.Z_CANVAS,  speed=0.2)
            try:
                input('press enter to move to next position')
            except SyntaxError:
                pass
            self._move(0,.35,self.Z_CANVAS+.09,  speed=0.2)
            # self.to_neutral(speed=0.2)
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
        # If you run painter.paint on a given x,y it will be slightly off (both in real and simulation)
        # Close this gap by transforming the given x,y into a coordinate that painter.paint will use
        # to perfectly hit that given x,y using a homograph transformation

        if use_cache and os.path.exists(os.path.join(self.opt.cache_dir, "cached_H_coord.pkl")):
            self.H_coord = pickle.load(open(os.path.join(self.opt.cache_dir, "cached_H_coord.pkl"),'rb'), encoding='latin1')
            return self.H_coord

        import matplotlib
        import matplotlib.pyplot as plt
        from scipy.ndimage import median_filter
        import cv2

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
            if not self.opt.ink:
                if i % 5 == 0: self.get_paint(0)
            i += 1
            x_prop, y_prop = canvas_coord # Coord in canvas proportions
            # x_pix, y_pix = int(x_prop * canvas_width_pix), int((1-y_prop) * canvas_height_pix) #  Coord in canvas pixels
            x_glob,y_glob,_ = canvas_to_global_coordinates(x_prop,y_prop,None, self.opt) # Coord in meters from robot

            # Paint the point
            dot_stroke = BrushStroke(self.opt).dot_stroke(self.opt)
            dot_stroke.execute(self, x_glob, y_glob, 0)

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
                window = canvas[max(0,y_pix-w):min(canvas_height_pix,y_pix+w), max(0,x_pix-w):min(x_pix+w, canvas_width_pix),:]
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



    def paint_extended_stroke_library(self, save_batch_size=10, image_save_height=768):
        w = self.opt.CANVAS_WIDTH_PIX
        h = self.opt.CANVAS_HEIGHT_PIX

        self.to_neutral()
        canvas_without_stroke = self.camera.get_canvas(max_height=image_save_height)
        strokes_without_getting_new_paint = 999 
        strokes_without_cleaning = 9999
        distance_since_getting_paint = 0

        lib_dir = os.path.join(self.opt.cache_dir, 'stroke_library')
        if not os.path.exists(lib_dir): os.mkdir(lib_dir)

        # Figure out how many strokes can be made on the given canvas size
        n_strokes_y = int(math.floor(self.opt.CANVAS_HEIGHT_M/(0.5*self.opt.MAX_STROKE_LENGTH)))

        brush_strokes = [] # list of BrushStroke
        canvases_before = [] # Photos of canvases without stroke
        canvases_after = [] # photos of canvases with stroke

        n_strokes = 0
        for paper_it in range(self.opt.num_papers):
            for y_offset_pix_og in np.linspace((.03/self.opt.CANVAS_HEIGHT_M), 0.99-(.02/self.opt.CANVAS_HEIGHT_M), n_strokes_y)*h:
                # for x_offset_pix in np.linspace(0.02, 0.99-(self.opt.MAX_STROKE_LENGTH/self.opt.CANVAS_WIDTH), n_strokes_x)*w:

                x_offset_pix = 0.02 * w 
                while(True): # x loop
                    if not self.opt.ink:
                        if strokes_without_cleaning >= 20:
                            self.clean_paint_brush()
                            self.get_paint(0)
                            strokes_without_cleaning, strokes_without_getting_new_paint = 0, 0
                            distance_since_getting_paint = 0
                        # if strokes_without_getting_new_paint >= 6:
                        if distance_since_getting_paint >= self.opt.max_length_before_new_paint:
                            self.get_paint(0)
                            strokes_without_getting_new_paint = 0
                            distance_since_getting_paint = 0
                        strokes_without_getting_new_paint += 1
                        strokes_without_cleaning += 1

                    random_stroke = BrushStroke(self.opt)
                    
                    stroke_length_m = random_stroke.get_path()[:,0].max().item()
                    stroke_length_pix = stroke_length_m * (w / self.opt.CANVAS_WIDTH_M)
                    if stroke_length_pix + x_offset_pix > 0.98*w:
                        break # No room left on the page width
                    
                    y_offset_pix = y_offset_pix_og

                    with torch.no_grad(): # Save other variables with brush stroke for training
                        random_stroke.xt *= 0
                        random_stroke.xt += (x_offset_pix / w)
                        random_stroke.yt *= 0
                        random_stroke.yt += (y_offset_pix / h)
                        random_stroke.a *= 0
                        
                    x, y = x_offset_pix / w, 1 - (y_offset_pix / h)
                    x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
                    x,y,_ = canvas_to_global_coordinates(x,y,None,self.opt)
                    
                    random_stroke.execute(self, x, y, 0)

                    distance_since_getting_paint += random_stroke.get_length().item()

                    self.to_neutral()
                    canvas_with_stroke = self.camera.get_canvas(max_height=image_save_height)

                    if n_strokes % 3 == 0:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt
                        fig = plt.figure()
                        ax = fig.gca()
                        # ax.scatter(canvas_with_stroke.shape[1]/2, canvas_with_stroke.shape[0]/2)

                        ax.imshow(canvas_with_stroke)

                        ax.scatter((x_offset_pix / w)*canvas_with_stroke.shape[1], 
                                   (y_offset_pix / h)*canvas_with_stroke.shape[0],
                                   facecolors='none', edgecolors='r')
                        ax.set_xticks([]), ax.set_yticks([])
                        fig.tight_layout()
                        self.writer.add_figure('stroke_library/{}'.format(n_strokes), fig, 0)
                        plt.close(fig)

                    brush_strokes.append(random_stroke)
                    canvases_before.append(canvas_without_stroke)
                    canvases_after.append(canvas_with_stroke)

                    n_strokes += 1

                    canvas_without_stroke = canvas_with_stroke.copy()

                    if n_strokes % save_batch_size == 0:
                        # Save data
                        bs_fn = os.path.join(lib_dir, 'stroke_parameters{:05d}.npy'.format(n_strokes))
                        with open(bs_fn,'wb') as f:
                            pickle.dump(brush_strokes, f)
                            
                        canvases_before_fn = os.path.join(lib_dir, 'canvases_before_{:05d}.npy'.format(n_strokes))
                        with gzip.GzipFile(canvases_before_fn, 'w') as f:
                            canvases_before = np.stack(np.stack(canvases_before, axis=0), axis=0)
                            canvases_before = (canvases_before * 255).astype(np.uint8)
                            np.save(f, canvases_before)
                        canvases_after_fn = os.path.join(lib_dir, 'canvases_after_{:05d}.npy'.format(n_strokes))
                        with gzip.GzipFile(canvases_after_fn, 'w') as f:
                            canvases_after = np.stack(np.stack(canvases_after, axis=0), axis=0)
                            canvases_after = (canvases_after * 255).astype(np.uint8)
                            np.save(f, canvases_after)
                            
                        with open(os.path.join(self.opt.cache_dir, 'stroke_library', 'stroke_settings_during_library.json'), 'w') as f:
                            settings = {}
                            settings['MAX_BEND'] = self.opt.MAX_BEND
                            settings['MIN_STROKE_Z'] = self.opt.MIN_STROKE_Z
                            settings['MIN_STROKE_LENGTH'] = self.opt.MIN_STROKE_LENGTH
                            settings['MAX_STROKE_LENGTH'] = self.opt.MAX_STROKE_LENGTH
                            settings['MAX_ALPHA'] = self.opt.MAX_ALPHA
                            settings['CANVAS_WIDTH_M'] = self.opt.CANVAS_WIDTH_M
                            settings['CANVAS_HEIGHT_M'] = self.opt.CANVAS_HEIGHT_M
                            json.dump(settings, f, indent=4)
                    
                        # Go back to empty because memory gets too large
                        brush_strokes = []
                        canvases_before = []
                        canvases_after = [] 
                    
                    gap = 0.03*w if self.opt.ink else 0.05*w
                    x_offset_pix += stroke_length_pix + gap

            if paper_it != self.opt.num_papers-1:
                try:
                    input('Place down new paper. Press enter to start.')
                except SyntaxError:
                    pass
                # Retake with the new paper
                canvas_without_stroke = self.camera.get_canvas(max_height=image_save_height)

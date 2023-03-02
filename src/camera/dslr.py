#! /usr/bin/env python3
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os 

# import camera.color_calib
from camera.color_calib import color_calib, find_calib_params
from camera.harris import find_corners
from camera.intrinsic_calib import computeIntrinsic
import glob

from camera.dslr_gphoto import *

# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

class WebCam():
    def __init__(self, opt, debug=False):
        self.camera = camera_init()
        self.debug = debug
        self.H_canvas = None

        self.has_color_info = False

        self.color_tmat = None
        self.greyval = None

        self.opt = opt

    def get_rgb_image(self, channels='rgb'):
        # while True:
        #     targ, img = capture_image(self.camera, channels, self.debug)
        #     plt.imshow(img)
        #     plt.show()

        return capture_image(self.camera, channels)
        
        # Dirty fix for image delay
        # for i in range(4):
        #     targ, img = capture_image(self.camera, channels)
        # return targ, img

    # return RGB image, color corrected
    def get_color_correct_image(self, use_cache=False):
        if not self.has_color_info:
            if not use_cache or not os.path.exists(os.path.join(self.opt.cache_dir, 'cached_color_calibration.pkl')):
                try:
                    input('No color info found. Beginning color calibration. Ensure you have placed Macbeth color checker in camera frame and press ENTER to continue.')
                except SyntaxError:
                    pass
                completed_color_calib = False
                while not completed_color_calib:
                    try:
                        self.init_color_calib()
                        retake = input("Retake? y/[n]")
                        if not(retake[:1] == 'y' or retake[:1] == 'Y'):
                            completed_color_calib = True
                    except Exception as e:
                        print(e)
                        try: input('could not calibrate. Move color checker and try again (press enter when ready)')
                        except SyntaxError: pass 
                try:    
                    input("Remove color checker from frame.")
                except SyntaxError:
                    pass
            else:
                params = pickle.load(open(os.path.join(self.opt.cache_dir, "cached_color_calibration.pkl"),'rb'), encoding='latin1')
                self.color_tmat, self.greyval = params["color_tmat"], params["greyval"]
                self.has_color_info = True

        path, img = self.get_rgb_image()

        # has to be done for some reason
        return cv2.cvtColor(color_calib(img, self.color_tmat, self.greyval), cv2.COLOR_BGR2RGB)

    def get_canvas(self, use_cache=False, max_height=2048):
        if self.H_canvas is None:
            self.calibrate_canvas(use_cache)
        
        # use corrected image if possible
        if (self.has_color_info):
            img = self.get_color_correct_image(use_cache)
        else:
            _, img = self.get_rgb_image()

        canvas = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
        w = int(img.shape[0] * (self.opt.CANVAS_WIDTH/self.opt.CANVAS_HEIGHT))
        canvas = canvas[:, :w]
        if max_height is not None and canvas.shape[0] > max_height:
            fact = 1.0 * img.shape[0] / max_height
            canvas = cv2.resize(canvas, (int(canvas.shape[1]/fact), int(canvas.shape[0]/fact)))
        return canvas

    def calibrate_canvas(self, use_cache=False):
        img = self.get_color_correct_image(use_cache=use_cache)
        h = img.shape[0]
        # original image shape is too wide of an aspect ratio compared to paper
        # w = int(h * LETTER_WH_RATIO)
        w = int(h * (self.opt.CANVAS_WIDTH/self.opt.CANVAS_HEIGHT))
        assert(w <= img.shape[1])

        if use_cache and os.path.exists(os.path.join(self.opt.cache_dir, 'cached_H_canvas.pkl')):
            self.H_canvas = pickle.load(open(os.path.join(self.opt.cache_dir, "cached_H_canvas.pkl"),'rb'), encoding='latin1')
            img1_warp = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
            # plt.imshow(img1_warp[:, :w])
            # plt.title('Hopefully this looks like just the canvas')
            # plt.show()
            return


        self.canvas_points = find_corners(img, show_search=self.debug)

        img_corners = img.copy()
        for corner_num in range(4):
            x, y = self.canvas_points[corner_num]

            # invert color to display
            for u in range(-10, 10):
                for v in range(-10, 10):
                    img_corners[y+u, x+v, :] = np.array((255, 255, 255)) - img_corners[y+u, x+v, :]

        plt.clf()
        plt.imshow(img_corners)
        plt.title("Here are the found corners")
        plt.show()

        true_points = np.array([[0,0],[w,0], [w,h],[0,h]])
        self.H_canvas, _ = cv2.findHomography(self.canvas_points, true_points)
        img1_warp = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
        
        # print(img1_warp[:, :w].shape)
        # print(img1_warp.shape)
        plt.imshow(img1_warp[:, :w])
        plt.title('Hopefully this looks like just the canvas')
        plt.show()
        # plt.imshow(img1_warp)
        # plt.title('Hopefully this looks like just the canvas')
        # plt.show()
        
        with open(os.path.join(self.opt.cache_dir, 'cached_H_canvas.pkl'),'wb') as f:
            pickle.dump(self.H_canvas, f)

    def init_color_calib(self):
        path, img = self.get_rgb_image()
        self.color_tmat, self.greyval = find_calib_params(path, self.debug)
        self.has_color_info = True
        
        with open(os.path.join(self.opt.cache_dir, 'cached_color_calibration.pkl'),'wb') as f:
            params = {"color_tmat":self.color_tmat, "greyval":self.greyval}
            pickle.dump(params, f)

    # intrinsic calibration of the camera
    def init_distortion_calib(self, imgs_exist=False, calib_path='./calibration/', num_imgs=10):
        # capture images if they do not exist
        if not imgs_exist:
            # capture set number of images, with i being editable to enable retaking
            i = 0
            while i < num_imgs:
                input("Maneuver checkerboard and press ENTER to capture image %d/%d." % ((i + 1), num_imgs))
                _, img = self.get_rgb_image()
                plt.imshow(img)
                plt.draw()
                plt.show(block=False)
                plt.pause(0.01)
                # retake if desired
                retake = input("Retake? y/[n]")
                plt.close()
                if retake[:1] == 'y' or retake[:1] == 'Y':
                    # do not save and do not increment
                    print("Retaking.")
                    continue
                else:
                    fname = calib_path + str(i).zfill(3) + ".jpg"
                    plt.imsave(fname, img)
                    print("Saved to " + fname + ".")
                    i += 1

        images = glob.glob(calib_path + "*.jpg")
        self.intrinsics = computeIntrinsic(images, (6, 8), (8, 8))
    
    # undistort and crop using OpenCV
    # From OpenCV tutorials
    def undistort(self, img):
        if self.intrinsics is None:
            input("No intrinsics matrix found. You must perform intrinsics calibration.")
            quit()
        # undistort
        mtx, dist, newCameraMtx, roi = self.intrinsics
        dst = cv2.undistort(img, mtx, dist, None, newCameraMtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        return dst

class SimulatedWebCam():
    def __init__(self, opt):
        self.opt = opt
        w_h_ratio = float(opt.CANVAS_WIDTH) / opt.CANVAS_HEIGHT
        h = 1024
        self.canvas = np.ones((h,int(h * w_h_ratio),3), dtype=np.float32) * 255.
    def get_canvas(self):
        return self.canvas
    def calibrate_canvas(self, use_cache=False):
        pass
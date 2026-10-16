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

import torch 
from torchvision.transforms import Resize

# import camera.color_calib
from camera.color_calib import color_calib, find_calib_params
from camera.harris import find_corners
from camera.intrinsic_calib import computeIntrinsic
import glob

from camera.dslr_gphoto import *

# Fixed rectified-canvas height (px). Width follows paper aspect from materials.
CANVAS_RENDER_HEIGHT = 768

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
        # Live-view preview by default (faster). Pass --high_quality_camera for full shutter.
        preview = not getattr(self.opt, 'high_quality_camera', False)
        return capture_image(self.camera, channels, preview=preview, debug=self.debug)

    # return RGB image, color corrected
    def get_color_correct_image(self, recalibrate=False):
        if not self.has_color_info and self.opt.calib_colors:
            color_calib_path = os.path.join(self.opt.cache_dir, 'cached_color_calibration.pkl')
            if recalibrate or not os.path.exists(color_calib_path):
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
                    input("Remove color checker from frame, then press enter.")
                except SyntaxError:
                    pass
            else:
                params = pickle.load(open(os.path.join(self.opt.cache_dir, "cached_color_calibration.pkl"),'rb'), encoding='latin1')
                self.color_tmat, self.greyval = params["color_tmat"], params["greyval"]
                self.has_color_info = True

        path, img = self.get_rgb_image()

        # has to be done for some reason
        if self.opt.calib_colors:
            return cv2.cvtColor(color_calib(img, self.color_tmat, self.greyval), cv2.COLOR_BGR2RGB)
        else:
            return img

    def _canvas_pixel_size(self):
        '''Fixed rectified canvas size (h, w) from paper aspect + CANVAS_RENDER_HEIGHT.

        Independent of camera frame aspect (preview vs shutter, portrait, etc.).
        '''
        # Options.__getattr__ KeyErrors missing keys, so don't use getattr(opt, ...).
        h = int(CANVAS_RENDER_HEIGHT)
        aspect = float(self.opt.CANVAS_WIDTH_M) / float(self.opt.CANVAS_HEIGHT_M)
        w = max(1, int(round(h * aspect)))
        return h, w

    def _load_h_canvas(self, path):
        raw = pickle.load(open(path, 'rb'), encoding='latin1')
        if isinstance(raw, dict) and 'H' in raw:
            self.H_canvas = raw['H']
            return raw.get('w'), raw.get('h')
        self.H_canvas = raw
        return None, None

    def _save_h_canvas(self, path, w, h):
        with open(path, 'wb') as f:
            pickle.dump({'H': self.H_canvas, 'w': w, 'h': h}, f)

    def get_canvas(self, recalibrate=False, max_height=None):
        if self.H_canvas is None:
            self.calibrate_canvas(recalibrate=recalibrate)
        
        # use corrected image if possible
        if (self.has_color_info and self.opt.calib_colors):
            img = self.get_color_correct_image(recalibrate=recalibrate)
        else:
            _, img = self.get_rgb_image()

        h, w = self._canvas_pixel_size()
        canvas = cv2.warpPerspective(img, self.H_canvas, (w, h))
        if max_height is not None and canvas.shape[0] > max_height:
            fact = 1.0 * canvas.shape[0] / max_height
            canvas = cv2.resize(canvas, (int(canvas.shape[1] / fact), int(canvas.shape[0] / fact)))
        return canvas
    
    def get_canvas_tensor(self, h=None, w=None):
        canvas = self.get_canvas()
        canvas = torch.from_numpy(canvas).permute(2,0,1).unsqueeze(0)
        if h is not None and w is not None:
            canvas = Resize((h,w), antialias=True)(canvas)

        canvas = torch.cat([canvas, torch.ones(1,1,canvas.shape[2],canvas.shape[3])], dim=1)

        return canvas

    def calibrate_canvas(self, recalibrate=False):
        img = self.get_color_correct_image(recalibrate=recalibrate)
        h, w = self._canvas_pixel_size()
        print(f'Canvas calib: image {img.shape[1]}x{img.shape[0]} -> '
              f'rectified canvas {w}x{h} (paper aspect '
              f'{self.opt.CANVAS_WIDTH_M / self.opt.CANVAS_HEIGHT_M:.3f})')

        canvas_h_path = os.path.join(self.opt.cache_dir, 'cached_H_canvas.pkl')
        if not recalibrate and os.path.exists(canvas_h_path):
            cached_w, cached_h = self._load_h_canvas(canvas_h_path)
            if cached_w is None:
                print('Warning: old cached_H_canvas.pkl has no size metadata; '
                      'warp target is now fixed 768px height. Use --recalibrate if it looks wrong.')
            elif (cached_w, cached_h) != (w, h):
                print(f'Warning: cached canvas size was {cached_w}x{cached_h}, '
                      f'now targeting {w}x{h}. Prefer --recalibrate if warp looks wrong.')
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

        true_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        self.H_canvas, _ = cv2.findHomography(self.canvas_points, true_points)
        img1_warp = cv2.warpPerspective(img, self.H_canvas, (w, h))

        plt.imshow(img1_warp)
        plt.title('Hopefully this looks like just the canvas')
        plt.show()

        self._save_h_canvas(canvas_h_path, w, h)

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
        w_h_ratio = float(opt.CANVAS_WIDTH_M) / opt.CANVAS_HEIGHT_M
        h = 1024
        self.canvas = np.ones((h,int(h * w_h_ratio),3), dtype=np.float32) * 255.
    def get_canvas(self):
        return self.canvas
    def get_canvas_tensor(self, h=None, w=None):
        canvas = self.get_canvas()
        canvas = torch.from_numpy(canvas).permute(2,0,1).unsqueeze(0)
        if h is not None and w is not None:
            canvas = Resize((h,w), antialias=True)(canvas)
        canvas = torch.cat([canvas, torch.ones(1,1,h,w)], dim=1)
        return canvas
    def calibrate_canvas(self, recalibrate=False):
        pass
#! /usr/bin/env python3
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dslr_gphoto as dslr

import color_calib
from harris import find_corners
from intrinsic_calib import computeIntrinsic
import glob

# Letter sized paper aspect ratio
LETTER_WH_RATIO = 279 / 216

# https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/opencv_viewer_example.py

class WebCam():
    def __init__(self):
        self.camera = dslr.camera_init()
        self.H_canvas = None

        self.has_color_info = False

        self.color_tmat = None
        self.greyval = None

    def get_rgb_image(self, channels='rgb'):
        return dslr.capture_image(self.camera, channels)

    # return RGB image, color corrected
    def get_color_correct_image(self):
        if not self.has_color_info:
            input("No color info found. Beginning color calibration. Ensure you have placed Macbeth color checker in camera frame and press ENTER to continue.")
            self.init_color_calib()
            input("Remove color checker from frame.")
        path, img = self.get_rgb_image()

        # has to be done for some reason
        return cv2.cvtColor(color_calib.color_calib(img, self.color_tmat, self.greyval), cv2.COLOR_BGR2RGB)

    def get_canvas(self):
        if self.H_canvas is None:
            self.calibrate_canvas()
        
        # use corrected image if possible
        if (self.has_color_info):
            img = self.get_color_correct_image()
        else:
            _, img = self.get_rgb_image()

        canvas = cv2.warpPerspective(img, self.H_canvas, (img.shape[1], img.shape[0]))
        return canvas

    def calibrate_canvas(self, show_search=False):
        path, img = self.get_rgb_image()
        h = img.shape[0]
        # original image shape is too wide of an aspect ratio compared to paper
        w = int(h * LETTER_WH_RATIO)
        assert(w <= img.shape[1])

        self.canvas_points = find_corners(img, 100, show_search)

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
        
        plt.imshow(img1_warp[:, :w])
        plt.title('Hopefully this looks like just the canvas')
        plt.show()

    def init_color_calib(self, disp_results=False):
        path, img = self.get_rgb_image()
        self.color_tmat, self.greyval = color_calib.find_calib_params(path, disp_results)
        self.has_color_info = True

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

    

    def test(self):
        img = self.get_rgb_image()
        d = self.get_depth_image()

        fig, ax = plt.subplots(1,2)
        ax[0].imshow(img)
        ax[1].imshow(d)
        plt.show()

def increase_brightness(img, value=30):
    # https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

cam = WebCam()
# cam.init_color_calib(True)
# img = cam.get_color_correct_image()
# plt.imshow(img)
# plt.show()
cam.calibrate_canvas()
# cam.init_distortion_calib()

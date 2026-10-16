# Capture image from DSLR camaera using gphoto2 Python interface
# Author: Jason Xu

##########################################################
#################### Copyright 2022 ######################
##################### by Jason Xu ########################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

# adapted from code at https://github.com/jim-easterbrook/python-gphoto2/blob/master/examples/capture-image.py

import glob
import os
import time

import cv2
import gphoto2 as gp
from matplotlib import pyplot as plt

# initialize camera object
def camera_init():
    # kill storage usb drivers
    os.system('pkill -f gphoto2')

    # initialize camera
    camera = gp.Camera()
    camera.init()
    camera.file_data = None

    return camera

def empty_event_queue(camera):
    while True:
        type_, data = camera.wait_for_event(10)
        if type_ == gp.GP_EVENT_TIMEOUT:
            return
        # if type_ == gp.GP_EVENT_FILE_ADDED:
        #     print()

def _prepare_capture_target():
    cap_dir = '/tmp/gphoto2_captures/'
    if not os.path.exists(cap_dir):
        os.mkdir(cap_dir)
    for f in glob.glob(os.path.join(cap_dir, '*')):
        os.remove(f)
    return os.path.join(cap_dir, str(time.time()) + '.jpg')

# capture image from camera object
# returns both the filename and numpy array of target
# preview=True uses live-view (faster); preview=False uses full shutter (higher quality)
def capture_image(camera, channels='rgb', preview=True, debug=False):
    if debug:
        mode = 'live preview' if preview else 'full shutter'
        print('Capturing image ({})'.format(mode))

    target = _prepare_capture_target()

    if preview:
        # Live view / preview frame — faster, still high enough quality for planning
        # https://github.com/jim-easterbrook/python-gphoto2/blob/master/examples/preview-image.py
        while True:
            try:
                camera_file = camera.capture_preview()
                camera_file.save(target)
                break
            except Exception:
                empty_event_queue(camera)
    else:
        # Full shutter capture — higher quality, slower
        while True:
            try:
                file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
                break
            except Exception:
                empty_event_queue(camera)

        if debug:
            print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
            print('Copying image to', target)
        camera_file = camera.file_get(
            file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
        camera_file.save(target)

    if channels == 'bgr':
        return target, cv2.imread(target)
    else:
        return target, plt.imread(target)

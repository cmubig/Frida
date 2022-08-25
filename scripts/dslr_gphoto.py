# Capture image from DSLR camaera using gphoto2 Python interface
# Author: Jason Xu

##########################################################
#################### Copyright 2022 ######################
##################### by Jason Xu ########################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

# adapted from code at https://github.com/jim-easterbrook/python-gphoto2/blob/master/examples/capture-image.py

import os
import gphoto2 as gp
from matplotlib import pyplot as plt
import cv2

# initialize camera object
def camera_init():
    # kill storage usb drivers
    os.system('pkill -f gphoto2')

    # initialize camera
    camera = gp.Camera()
    camera.init()

    return camera

# capture image from camera object
# returns both the filename and numpy array of target
def capture_image(camera, channels='rgb', debug=False):
    if (debug):
        print('Capturing image')
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    # file_path = camera.capture(gp.GP_CAMERA_CAPTURE_PREVIEW)
    # https://github.com/jim-easterbrook/python-gphoto2/blob/master/examples/preview-image.py
    # print((gp.gp_camera_capture_preview(camera)))
    # file_path = gp.gp_camera_capture_preview(camera)[1]

    if (debug):
        print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    
    # target = os.path.join('/tmp', file_path.name)
    import time 
    import glob
    cap_dir = '/tmp/gphoto2_captures/'
    if not os.path.exists(cap_dir): os.mkdir(cap_dir)
    files_in_cap_dir = glob.glob(os.path.join(cap_dir, '*'))
    for f in files_in_cap_dir:
        os.remove(f)
    target = os.path.join(cap_dir, str(time.time()) + '.jpg')

    if (debug):
        print('Copying image to', target)
    camera_file = camera.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)
    
    if channels=='bgr':
        return target, cv2.imread(target)
    else:
        return target, plt.imread(target)

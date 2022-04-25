# Capture image from DSLR camaera using gphoto2 Python interface
# Author: Jason Xu

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
    if (debug):
        print('Camera file path: {0}/{1}'.format(file_path.folder, file_path.name))
    target = os.path.join('/tmp', file_path.name)
    if (debug):
        print('Copying image to', target)
    camera_file = camera.file_get(
        file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)
    
    if channels=='bgr':
        return target, cv2.imread(target)
    else:
        return target, plt.imread(target)

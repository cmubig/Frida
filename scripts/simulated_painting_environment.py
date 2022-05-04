#! /usr/bin/env python3

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import numpy as np
import cv2
import math
import copy
from scipy import ndimage

from paint_utils import show_img

# Save some processed strokes to save time
# {stroke_ind+'_'+rotation+'_'+stroke_size: stroke nd.array}}
processed_stroke_cache = {}
processed_s_expanded_cache = {}

def apply_stroke(canvas, stroke, stroke_ind, color, x, y, theta=0):
    '''
    args:
        canvas (np.array[width, height, 3]) : Current painting canvas 0-1 RGB
        stroke (np.array[width, height]) :  Stroke 0-1 intesity map
        color (np.array[3]) : RGB color to use for the brush stroke
        x, y (int) : location to apply the stroke in pixels. The start of the brush stroke.
        theta (float) : degrees of rotation
    '''
    # how to get to the start of the brush stroke from the top left of the cut out region
    down = 0.5
    right = 0.2

    stroke_cache_key = str(stroke_ind)+'_'+str(theta)+'_'+str(stroke.shape[0])

    if stroke_cache_key in processed_stroke_cache:
        stroke = processed_stroke_cache[stroke_cache_key].copy()
        s_expanded = processed_s_expanded_cache[stroke_cache_key].copy()
    else:
        # Padding for rotation. Ensure start of brush stroke is centered in square image
        h, w = stroke.shape
        padX = int((1 - right - right)*w), 0
        padY = int((1 - right)*w - (1 - down)*h), int((1 - right)*w - (1 - down)*h)
        imgP = np.pad(stroke, [padY, padX], 'constant')
        # show_img(imgP)

        # Rotate theta degrees
        stroke = ndimage.rotate(imgP, theta, reshape=False)
        # show_img(stroke)

        processed_stroke_cache[stroke_cache_key] = stroke.copy()

        s_expanded = np.tile(stroke[:,:, np.newaxis], (1,1,3))
        processed_s_expanded_cache[stroke_cache_key] = s_expanded.copy()

    s_color = s_expanded * color[None, None, :]
    stroke_dim = stroke.shape

    h, w = stroke.shape # new height and width (with padding. should be equal)

    # Get indices of the canvas to apply the stroke to
    y_s, y_e = max(y - int(.5 * h), 0), min(y - int(.5 * h) + stroke_dim[0], canvas.shape[0])
    x_s, x_e = max(x - int(.5 * w), 0), min(x - int(.5 * w) + stroke_dim[1], canvas.shape[1])

    # Get the indices of the stroke to apply to the canvas (incase stroke falls off side of canvas)
    sy_s = max(0, -1 * (y - int(.5 * h)))
    sy_e = y_e - y_s - min(0, (y - int(.5 * h)))
    sx_s = max(0, -1 * (x - int(.5 * w)))
    sx_e = x_e - x_s - min(0, (x - int(.5 * w)))
#     print(y_s, y_e, sy_s, sy_e)
#     print(x_s, x_e, sx_s, sx_e)
    # Apply the stroke to the canvas
    canvas[y_s:y_e,x_s:x_e,:] \
        = canvas[y_s:y_e,x_s:x_e,:] * (1 - s_expanded[sy_s:sy_e,sx_s:sx_e]) + s_color[sy_s:sy_e,sx_s:sx_e]

    # plt.imshow(s_expanded)
    # plt.show()
    stroke_bool_map = np.zeros((canvas.shape[0], canvas.shape[1]))
    stroke_bool_map[y_s:y_e,x_s:x_e] = s_expanded[sy_s:sy_e,sx_s:sx_e,0]

    # plt.imshow(stroke_bool_map)
    # plt.colorbar()
    # plt.show()
    bbox = y_s, y_e, x_s, x_e
    return canvas, stroke_bool_map, bbox


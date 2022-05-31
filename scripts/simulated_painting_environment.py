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
down_cache = {}
right_cache = {}


def crop_stroke(stroke, down, right):
    # Remove black space from stroke map
    stroke_bool = stroke > 0.3
    
    # show_img(stroke_bool)
    all_black_cols = np.max(stroke_bool, axis=0) > 0.3

    last_filled_col = len(all_black_cols) - np.argmax(all_black_cols[::-1])
    first_filled_col = np.argmax(all_black_cols)


    all_black_rows = np.max(stroke_bool, axis=1) > 0.3

    last_filled_row = len(all_black_rows) - np.argmax(all_black_rows[::-1])
    first_filled_row = np.argmax(all_black_rows)
    
    h,w = stroke.shape

    # padding so it's not perfectly cropped
    pw = max(5,int(0.1 * (last_filled_col-first_filled_col)))
    ph = max(5,int(0.1 * (last_filled_row-first_filled_row)))
    last_filled_col = min(w, last_filled_col + pw)
    last_filled_row = min(h, last_filled_row + ph)
    first_filled_col = max(0, first_filled_col - pw)
    first_filled_row = max(0, first_filled_row - ph)


    a = down*h - first_filled_row
    b = (1-down)*h - (h-last_filled_row)
    new_down = a / (a+b)
    # print(a, b, new_down, down, h)
    a = right*w - first_filled_col
    b = (1-right)*w - (w-last_filled_col)
    new_right = a / (a + b)

    t = 0
    return stroke[max(0,first_filled_row-t):last_filled_row+t, \
                  max(0,first_filled_col-t):last_filled_col+t], \
            new_down, new_right

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
        stroke = processed_stroke_cache[stroke_cache_key]#.copy()
        s_expanded = processed_s_expanded_cache[stroke_cache_key].copy()
        right = right_cache[stroke_ind]
        down = down_cache[stroke_ind]
    else:
        # Crop the stroke. The smaller it is the faster this will be
        # show_img(stroke)
        # import matplotlib.pyplot as plt 
        # plt.imshow(stroke)
        # plt.scatter(right*stroke.shape[1], (down)*stroke.shape[0])
        # plt.show()

        # a, b, c = crop_stroke(stroke, down, right)
        # plt.imshow(a)
        # plt.scatter(c*a.shape[1], (b)*a.shape[0])
        # plt.show()
        stroke, down, right = crop_stroke(stroke.copy(), down, right)
        
        down_cache[stroke_ind] = down 
        right_cache[stroke_ind] = right


        # Padding for rotation. Ensure start of brush stroke is centered in square image
        h, w = stroke.shape

        # PadX to center the start of stroke
        padX = max(0, w*(1-2*right)), max(0, w*(2*right-1))
        # PadY to center the start of the stroke
        padY = max(0, h*(1-2*down)), max(0, h*(2*down - 1))
        # print(padX, padY)
        # Pad to become square
        newW, newH = padX[0] + padX[1] + w, padY[0] + padY[1] + h 
        # print(newW, newH)
        if newH > newW:
            xtra_x = (newH - newW)/2 
            padX = padX[0] + xtra_x, padX[1] + xtra_x
        elif newH < newW:
            xtra_y = (newW - newH)/2 
            padY = padY[0] + xtra_y, padY[1] + xtra_y
        padX = int(padX[0]), int(padX[1])
        padY = int(padY[0]), int(padY[1])

        # print(padX, padY, right, down, h, w, 'should only happen in beginning')
        imgP = np.pad(stroke, [padY, padX], 'constant')
        # show_img(imgP)
        # print(imgP.shape)

        # Rotate theta degrees
        stroke = ndimage.rotate(imgP, theta, reshape=False)
        # show_img(stroke)

        # processed_stroke_cache[stroke_cache_key] = stroke.copy()

        s_expanded = np.tile(stroke[:,:, np.newaxis], (1,1,3))
        # processed_s_expanded_cache[stroke_cache_key] = s_expanded.copy()

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
    # print(y_s, y_e, sy_s, sy_e)
    # print(x_s, x_e, sx_s, sx_e)

    # Apply the stroke to the canvas
    canvas[y_s:y_e,x_s:x_e,:] \
        = canvas[y_s:y_e,x_s:x_e,:] * (1 - s_expanded[sy_s:sy_e,sx_s:sx_e]) + s_color[sy_s:sy_e,sx_s:sx_e]

    stroke_bool_map = np.zeros((canvas.shape[0], canvas.shape[1]), dtype=np.bool)
    stroke_bool_map[y_s:y_e,x_s:x_e] = s_expanded[sy_s:sy_e,sx_s:sx_e,0] > 0.2

    bbox = y_s, y_e, x_s, x_e
    return canvas, stroke_bool_map, bbox

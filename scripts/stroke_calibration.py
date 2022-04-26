#! /usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from scipy import ndimage

import requests
import PIL.Image
from io import BytesIO

from painter import *
from strokes import *
from simulated_painting_environment import apply_stroke


"""
Some Helper Functions
"""

def show_img(img):
    # Display at actual size: https://stackoverflow.com/questions/60144693/show-image-in-its-original-resolution-in-jupyter-notebook
    # Acquire default dots per inch value of matplotlib
    dpi = matplotlib.rcParams['figure.dpi']
    # Determine the figures size in inches to fit your image
    height, width = img.shape[0], img.shape[1]
    figsize = width / float(dpi), height / float(dpi)

    plt.figure(figsize=figsize)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

# def pil_loader_internet(url):
#     response = requests.get(url)
#     img = PIL.Image.open(BytesIO(response.content))
#     return img.convert('RGB')

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


def process_stroke_library(raw_strk_lib):
    """
    The input is a raw photo of the canvas after creating the stroke library

    """
    raw_strk_lib = increase_brightness(raw_strk_lib)
    raw_strk_lib = cv2.cvtColor(raw_strk_lib,cv2.COLOR_BGR2GRAY)
    # raw_strk_lib = cv2.resize(raw_strk_lib, (int(raw_strk_lib.shape[1]/6),int(raw_strk_lib.shape[0]/6)))
    # show_img(raw_strk_lib)

    strokes = []

    canvas = np.ones((raw_strk_lib.shape[0], raw_strk_lib.shape[1], 3)) * 255.

    # Cut out and process the strokes from the stroke library picture
    i = 0
    np.random.seed(0)
    # for x_start in range(0,CANVAS_WIDTH, cell_dim_x):
    #     for y_start in range(0, CANVAS_HEIGHT, cell_dim_y):
    for x_start in np.linspace(0,CANVAS_WIDTH, num=cells_x+1)[:-1]:
        for y_start in np.linspace(0, CANVAS_HEIGHT, num=cells_y+1)[:-1]:
            i += 1
            if i > len(all_strokes): break
                
            # Bounding box indices of the stroke in the picture of stroke library
            x_start_pix = int(math.floor((x_start/CANVAS_WIDTH)*raw_strk_lib.shape[1]))
            x_end_pix = int(math.floor(((x_start + cell_dim_x)/CANVAS_WIDTH)*raw_strk_lib.shape[1]))
            y_start_pix = int(math.floor((y_start/CANVAS_HEIGHT)*raw_strk_lib.shape[0]))
            y_end_pix = int(math.floor(((y_start + cell_dim_y)/CANVAS_HEIGHT)*raw_strk_lib.shape[0]))
            
            # print(x_start_pix, x_end_pix)
            # Get just the single stroke
            single_strk_img = raw_strk_lib[y_start_pix:y_end_pix,x_start_pix:x_end_pix]
            # show_img(single_strk_img)
            
            # Make the background purely white
            stroke = single_strk_img.copy()
            stroke[stroke > 170] = 255
            # show_img(stroke)

            # Edges should be white
            ep = 0.1
            stroke[:int(ep*stroke.shape[0]),:] = 255
            stroke[int(stroke.shape[0] - ep*stroke.shape[0]):,:] = 255
            stroke[:,:int(ep*stroke.shape[1])] = 255
            stroke[:,int(stroke.shape[1] - ep*stroke.shape[1]):] = 255
            
            if stroke.min() > 100: continue # No paint
            
            # Convert to 0-1 where 1 is the stroke
            stroke = stroke/255.
            stroke = 1 - stroke # background 0, paint 1-ish
            stroke = stroke**.5 # Make big values bigger. Makes paint more opaque
            stroke /= stroke.max() # Max should be 1.0
            # show_img(stroke)
            
            # Color the stroke
            color = np.random.randint(0,high=255,size=3)
        
            x, y = np.random.randint(0,300), np.random.randint(0,300)
            #canvas,_ = apply_stroke(canvas, stroke, color, x, y)
            #canvas,_ = apply_stroke(canvas, stroke, color, x_start_pix, y_start_pix)
            canvas,_ = apply_stroke(canvas, stroke, len(strokes), color, 
                                  x_start_pix+int(0.2*stroke.shape[1]), y_start_pix+int(0.5*stroke.shape[0]), -12)
            strokes.append(stroke)
    show_img(canvas/255.)
    return strokes
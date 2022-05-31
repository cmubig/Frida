#! /usr/bin/env python3

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

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
from paint_utils import show_img, increase_brightness
from simulated_painting_environment import apply_stroke


"""
Some Helper Functions
"""


# def pil_loader_internet(url):
#     response = requests.get(url)
#     img = PIL.Image.open(BytesIO(response.content))
#     return img.convert('RGB')



def process_stroke_library(raw_strk_lib, opt):
    """
    The input is a raw photo of the canvas after creating the stroke library
    cut out the brush strokes, process them and return them as an array
    """
    raw_strk_lib = increase_brightness(raw_strk_lib)
    raw_strk_lib = cv2.cvtColor(raw_strk_lib,cv2.COLOR_BGR2GRAY)
    # raw_strk_lib = cv2.resize(raw_strk_lib, (int(raw_strk_lib.shape[1]/6),int(raw_strk_lib.shape[0]/6)))
    # show_img(raw_strk_lib)

    strokes = []

    canvas = np.ones((raw_strk_lib.shape[0], raw_strk_lib.shape[1], 3)) * 255.

    # Cut out and process the strokes from the stroke library picture
    i = 0
    stroke_ind = 0
    np.random.seed(0)
    # for x_start in range(0,opt.CANVAS_WIDTH, opt.cell_dim_x):
    #     for y_start in range(0, opt.CANVAS_HEIGHT, opt.cell_dim_y):
    forbidden = ((0,0), (0,opt.cells_x-1), (opt.cells_y-1,0), (opt.cells_y-1, opt.cells_x-1))

    for x_start in np.linspace(0,opt.CANVAS_WIDTH, num=opt.cells_x+1)[:-1]:
        for y_start in np.linspace(0, opt.CANVAS_HEIGHT, num=opt.cells_y+1)[:-1]:
            if (i%opt.cells_y,int(np.floor(i/(opt.cells_x+1)))) in forbidden: 
                i+=1
                continue
            i+=1
            stroke_ind += 1
            if stroke_ind > len(all_strokes): break
                
            # Bounding box indices of the stroke in the picture of stroke library
            x_start_pix = int(math.floor((x_start/opt.CANVAS_WIDTH)*raw_strk_lib.shape[1]))
            x_end_pix = int(math.floor(((x_start + opt.cell_dim_x)/opt.CANVAS_WIDTH)*raw_strk_lib.shape[1]))
            y_start_pix = int(math.floor((y_start/opt.CANVAS_HEIGHT)*raw_strk_lib.shape[0]))
            y_end_pix = int(math.floor(((y_start + opt.cell_dim_y)/opt.CANVAS_HEIGHT)*raw_strk_lib.shape[0]))
            
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
            # stroke = stroke**.5 # Make big values bigger. Makes paint more opaque
            stroke /= stroke.max() # Max should be 1.0
            # show_img(stroke)
            
            # Color the stroke
            color = np.random.randint(0,high=255,size=3)
        
            x, y = np.random.randint(0,300), np.random.randint(0,300)
            #canvas,_,_ = apply_stroke(canvas, stroke, color, x, y)
            #canvas,_,_ = apply_stroke(canvas, stroke, color, x_start_pix, y_start_pix)
            canvas,_,_ = apply_stroke(canvas, stroke, len(strokes), color, 
                                  x_start_pix+int(0.2*stroke.shape[1]), y_start_pix+int(0.5*stroke.shape[0]), -12)
            strokes.append(stroke)
    show_img(canvas/255.)
    return strokes

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
import pickle
import os
import time
import sys
import subprocess
from tqdm import tqdm
from scipy.ndimage import median_filter
from PIL import Image

from simulated_painting_environment import apply_stroke
from painter import canvas_to_global_coordinates
from strokes import all_strokes, paint_diffvg, simple_parameterization_to_real, StrokeBD
from paint_utils import *

def parse_csv_line(line, painter, colors):
    toks = line.split(',')
    if len(toks) != 7:
        return None
    x = int(float(toks[0])*painter.opt.CANVAS_WIDTH_PIX)
    y = int(float(toks[1])*painter.opt.CANVAS_HEIGHT_PIX)
    r = float(toks[2])*(360/(2*3.14))
    stroke_ind = int(toks[3])
    color = np.array([float(toks[4]), float(toks[5]), float(toks[6])])*255.
    color_ind, color_discrete = nearest_color(color, colors)

    return x, y, r, stroke_ind, color, color_ind, color_discrete



def parse_csv_line_continuous(line, painter, colors):
    toks = line.split(',')
    if len(toks) != 9:
        return None
    x = int(float(toks[0])*painter.opt.CANVAS_WIDTH_PIX)
    y = int(float(toks[1])*painter.opt.CANVAS_HEIGHT_PIX)
    r = float(toks[2])*(360/(2*3.14))
    # stroke_ind = int(toks[3])
    length = float(toks[3])
    thickness = float(toks[4])
    bend = float(toks[5])
    color = np.array([float(toks[6]), float(toks[7]), float(toks[8])])*255.
    color_ind, color_discrete = nearest_color(color, colors)


    return x, y, r, length, thickness, bend, color, color_ind, color_discrete

def get_real_colors(painter, colors):
    painter.to_neutral()
    canvas_before = painter.camera.get_canvas()

    for color_ind in range(len(colors)):
        painter.clean_paint_brush()
        painter.get_paint(color_ind)
        for j in range(2):
            # Convert the canvas proportion coordinates to meters from robot
            x, y = color_ind*(0.9/len(colors)) + 0.1, 0.1 + j*(0.9/4)
            x,y,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

            # Paint the brush stroke
            # all_strokes[-1]().paint(painter, x, y, 45)
            StrokeBD().paint(painter, x, y, 45)

        painter.to_neutral()
        canvas_after = painter.camera.get_canvas()

        # Update representation of paint color
        new_paint_color = extract_paint_color(canvas_before, canvas_after, None)
        color_momentum = 1.0
        if new_paint_color is not None:
            colors[color_ind] = colors[color_ind] * (1-color_momentum) \
                            + new_paint_color * color_momentum
        else:
            colors[color_ind] = colors[color_ind] * 0 + 230 # Basically white
        canvas_before = canvas_after
    return colors

def save_for_python3_file(painter, full_sim_canvas):
    # Save strokes for planning python file
    painter.to_neutral()
    if painter.opt.simulate:
        current_canvas = full_sim_canvas.astype(np.uint8) 
    else:
        current_canvas = painter.camera.get_canvas().astype(np.uint8) 
    im = Image.fromarray(current_canvas)
    im.save(os.path.join(painter.opt.cache_dir, 'current_canvas.jpg'))

    # im = Image.fromarray(target.astype(np.uint8))
    # im.save(os.path.join(painter.opt.cache_dir, 'target_discrete.jpg'))

    # Save paint colors for the other python file
    # with open(os.path.join(painter.opt.cache_dir, 'colors.npy'), 'wb') as f:
    #     np.save(f, colors)

def paint_color_calibration(painter, colors):
    ''' Make a mark with each paint color, then measure that color '''
    if not painter.opt.simulate:
        cached_color_fn = os.path.join(painter.opt.cache_dir, 'colors.npy')
        if not painter.opt.use_cached_colors or not os.path.exists(cached_color_fn):
            try:
                input('About to measure paint colors. put blank page down then press enter.')
            except SyntaxError:
                pass
            colors = get_real_colors(painter, colors)
            all_colors = save_colors(colors)
            show_img(all_colors/255., title="Actual Measured Colors")
            try:
                input('Make sure blank canvas is exposed. Press enter to start painting.')
            except SyntaxError:
                pass
        else:
            colors = pickle.load(open(cached_color_fn,'rb'))
    return colors

global_it = 0
def paint_planner_new(painter, how_often_to_get_paint=5):
    global global_it
    painter.to_neutral()
    canvas_after = painter.camera.get_canvas()
    full_sim_canvas = canvas_after.copy()
    real_canvases = [canvas_after]
    consecutive_paints = 0
    consecutive_strokes_no_clean = 0
    camera_capture_interval = 8
    curr_color = -1

    # colors = paint_color_calibration(painter, colors)

    # target = discretize_with_labels(colors, labels)

    for it in tqdm(range(100 if painter.opt.adaptive else 1)): # how many times to go visit planning file
        canvas_before = canvas_after 

        # Save data that the python3 file needs
        save_for_python3_file(painter, full_sim_canvas)

        # Plan the new strokes with SawyerPainter/src/plan.py
        if not painter.opt.dont_plan or (painter.opt.adaptive and it>0):
            add_adapt = ['--generate_whole_plan'] if painter.opt.adaptive and it==0 else []
            
            try:
                import rospkg
                rospack = rospkg.RosPack()
                # get the file path for painter code
                ros_dir = rospack.get_path('paint')
            except: # Not running in ROS
                ros_dir = ''

            exit_code = subprocess.call(['python3', 
                os.path.join(ros_dir, 'src', 'plan.py')]+sys.argv[1:]+['--global_it', str(global_it)]+add_adapt)
            if exit_code != 0:
                print('exit code', exit_code)
                return
        
        # Colors possibly updated during planning
        # if painter.opt.prompt is not None:
        #if not opt.use_colors_from:
        with open(os.path.join(painter.opt.cache_dir, 'colors_updated.npy'), 'rb') as f:
            colors = np.load(f).astype(np.float32)
        all_colors = save_colors(colors)
        painter.writer.add_image('paint_colors/updated', all_colors/255., it)
        painter.writer.add_image('paint_colors/updated', all_colors/255., it)

        if not painter.opt.simulate and it == 0:
            show_img(canvas_before/255., title="Ready to start painting. Ensure mixed paint is provided and then exit this to start painting.")


        # Run Planned Strokes
        with open(os.path.join(painter.opt.cache_dir, "next_brush_strokes.csv"), 'r') as fp:
            instructions = [parse_csv_line_continuous(line, painter, colors) for line in fp.readlines()] 
            #max_instructions = 40
            n_instr = len(instructions)
            if painter.opt.adaptive:
                instructions = instructions[:painter.opt.strokes_before_adapting]
            for instruction in tqdm(instructions[:], desc="Painting"):
                canvas_before = canvas_after
                
                x, y, r, length, thickness, bend, color, color_ind, color_discrete = instruction
                
                color = colors[color_ind].copy()
                if painter.opt.simulate:
                    # Add some noise
                    x = x + np.random.randint(10)-5
                    y = y + np.random.randint(10)-5
                    r = r + np.random.randn(1)*10
                    color += np.random.randn(3)*5

                # Clean paint brush and/or get more paint
                new_paint_color = color_ind != curr_color
                if new_paint_color or consecutive_strokes_no_clean > 12:
                    dark_to_light = np.mean(colors[curr_color]) < np.mean(colors[color_ind])
                    # if dark_to_light and curr_color != -1:
                    #     painter.clean_paint_brush() # Really clean this thing
                    #     painter.clean_paint_brush()
                    #     if not painter.opt.simulate:
                    #         show_img(target/255., title="About to start painting with a lighter color")
                    painter.clean_paint_brush()
                    if consecutive_strokes_no_clean <= 12: painter.clean_paint_brush()
                    consecutive_strokes_no_clean = 0
                    curr_color = color_ind
                    new_paint_color = True
                # else:
                #     consecutive_strokes_no_clean += 1
                if consecutive_paints >= how_often_to_get_paint or new_paint_color:
                    painter.get_paint(color_ind)
                    consecutive_paints = 0

                # Convert the canvas proportion coordinates to meters from robot
                x, y = float(x) / painter.opt.CANVAS_WIDTH_PIX, 1 - (float(y) / painter.opt.CANVAS_HEIGHT_PIX)
                x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
                x,y,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

                # Paint the brush stroke
                s = simple_parameterization_to_real(length, bend, thickness)
                s.paint(painter, x, y, r * (2*3.14/360))

                if global_it%camera_capture_interval == 0:
                    painter.to_neutral()
                    canvas_after = painter.camera.get_canvas() if not painter.opt.simulate else full_sim_canvas
                    
                    real_canvases.append(canvas_after)

                    # # Update representation of paint color
                    # new_paint_color = extract_paint_color(canvas_before, canvas_after, None)
                    # color_momentum = 0.2
                    # if new_paint_color is not None:
                    #     colors[color_ind] = colors[color_ind] * (1-color_momentum) \
                    #                       + new_paint_color * color_momentum
                    # target = discretize_with_labels(colors, labels)

                if global_it%5==0: # loggin be sloggin
                    if not painter.opt.simulate:
                        painter.writer.add_image('images/canvas', canvas_after/255., global_it)
                    all_colors = save_colors(colors)
                    painter.writer.add_image('paint_colors/are', all_colors/255., global_it)
                    # painter.writer.add_image('target/target_discrete', target/255., global_it)

                global_it += 1
                consecutive_paints += 1

                # if global_it % 100 == 0:
                #     # to_gif(all_canvases)
                #     to_video(real_canvases, fn='real_canvases.mp4')
            if painter.opt.adaptive:
                if n_instr <= painter.opt.strokes_before_adapting:
                    break


    painter.clean_paint_brush()
    painter.clean_paint_brush()
    painter.clean_paint_brush()
    painter.clean_paint_brush()
    to_video(real_canvases, fn=os.path.join(painter.opt.plan_gif_dir,'real_canvases{}.mp4'.format(str(time.time()))))

    canvas_after = painter.camera.get_canvas() if not painter.opt.simulate else full_sim_canvas
    if not painter.opt.simulate:
        painter.writer.add_image('images/canvas', canvas_after/255., global_it)



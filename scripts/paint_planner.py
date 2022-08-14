
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
from strokes import all_strokes, paint_diffvg, simple_parameterization_to_real
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
            all_strokes[-1]().paint(painter, x, y, 45)

        painter.to_neutral()
        canvas_after = painter.camera.get_canvas()

        # Update representation of paint color
        new_paint_color = extract_paint_color(canvas_before, canvas_after, None)
        color_momentum = 1.0
        if new_paint_color is not None:
            colors[color_ind] = colors[color_ind] * (1-color_momentum) \
                            + new_paint_color * color_momentum
        canvas_before = canvas_after
    return colors

def save_for_python3_file(painter, colors, target, full_sim_canvas):
    # Save strokes for planning python file
    painter.to_neutral()
    if painter.opt.simulate:
        current_canvas = full_sim_canvas.astype(np.uint8) 
    else:
        current_canvas = painter.camera.get_canvas().astype(np.uint8) 
    im = Image.fromarray(current_canvas)
    im.save(os.path.join(painter.opt.cache_dir, 'current_canvas.jpg'))

    im = Image.fromarray(target.astype(np.uint8))
    im.save(os.path.join(painter.opt.cache_dir, 'target_discrete.jpg'))

    # Save paint colors for the other python file
    with open(os.path.join(painter.opt.cache_dir, 'colors.npy'), 'wb') as f:
        np.save(f, colors)

def paint_color_calibration(painter):
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
            # show_img(all_colors/255., title="Actual Measured Colors")
            try:
                input('Make sure blank canvas is exposed. Press enter to start painting.')
            except SyntaxError:
                pass
        else:
            colors = pickle.load(open(cached_color_fn,'rb'))
    return colors

global_it = 0
def paint_planner_new(painter, target, colors, labels, how_often_to_get_paint=5):
    global global_it
    painter.to_neutral()
    canvas_after = painter.camera.get_canvas()
    full_sim_canvas = canvas_after.copy()
    real_canvases = [canvas_after]
    sim_canvases = [canvas_after]
    consecutive_paints = 0
    camera_capture_interval = 8
    curr_color = -1

    # colors = paint_color_calibration(painter)

    target = discretize_with_labels(colors, labels)

    for it in tqdm(range(100 if painter.opt.adaptive else 1)): # how many times to go visit planning file
        canvas_before = canvas_after 

        # Save data that the python3 file needs
        save_for_python3_file(painter, colors, target, full_sim_canvas)

        # Plan the new strokes with SawyerPainter/scripts/plan_all_strokes.py
        if not painter.opt.dont_plan:
            add_adapt = ['--generate_whole_plan'] if painter.opt.adaptive and it==0 else []
            exit_code = subprocess.call(['python3', '/home/frida/ros_ws/src/intera_sdk/SawyerPainter/scripts/plan_all_strokes.py']+sys.argv[1:]+['--global_it', str(global_it)]+add_adapt)
            if exit_code != 0:
                print('exit code', exit_code)
                return
        if not painter.opt.simulate:
            show_img(canvas_before/255., title="Ready to start painting. Ensure mixed paint is provided and then exit this to start painting.")

        # Run Planned Strokes
        with open(os.path.join(painter.opt.cache_dir, "next_brush_strokes.csv"), 'r') as fp:
            if painter.opt.discrete:
                instructions = [parse_csv_line(line, painter, colors) for line in fp.readlines()] 
            else:
                instructions = [parse_csv_line_continuous(line, painter, colors) for line in fp.readlines()] 
            #max_instructions = 40
            if painter.opt.adaptive:
                instructions = instructions[:painter.opt.strokes_before_adapting]
            for instruction in tqdm(instructions[:]):
                canvas_before = canvas_after
                if painter.opt.discrete:
                    x, y, r, stroke_ind, color, color_ind, color_discrete = instruction
                else:
                    x, y, r, length, thickness, bend, color, color_ind, color_discrete = instruction
                color = colors[color_ind].copy()
                if painter.opt.simulate:
                    # Add some noise
                    x = x + np.random.randint(10)-5
                    y = y + np.random.randint(10)-5
                    r = r + np.random.randn(1)*10
                    color += np.random.randn(3)*5

                # Make the brush stroke on a simulated canvas
                try:
                    if painter.opt.discrete:
                        full_sim_canvas, _, _ = apply_stroke(full_sim_canvas, painter.strokes[stroke_ind], stroke_ind,
                            color, x, y, r)
                except Exception as e:
                    print('exception applying stroke', e)

                # Clean paint brush and/or get more paint
                new_paint_color = color_ind != curr_color
                if new_paint_color:
                    dark_to_light = np.mean(colors[curr_color]) < np.mean(colors[color_ind])
                    if dark_to_light and curr_color != -1:
                        painter.clean_paint_brush() # Really clean this thing
                        painter.clean_paint_brush()
                        if not painter.opt.simulate:
                            show_img(target/255., title="About to start painting with a lighter color")
                    painter.clean_paint_brush()
                    painter.clean_paint_brush()
                    curr_color = color_ind
                if consecutive_paints >= how_often_to_get_paint or new_paint_color:
                    painter.get_paint(color_ind)
                    consecutive_paints = 0

                # Convert the canvas proportion coordinates to meters from robot
                x, y = float(x) / painter.opt.CANVAS_WIDTH_PIX, 1 - (float(y) / painter.opt.CANVAS_HEIGHT_PIX)
                x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
                x,y,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

                # Paint the brush stroke
                if painter.opt.discrete:
                    all_strokes[stroke_ind]().paint(painter, x, y, r * (2*3.14/360))
                else:
                    s = simple_parameterization_to_real(length, bend, thickness)
                    s.paint(painter, x, y, r * (2*3.14/360))

                if global_it%camera_capture_interval == 0:
                    painter.to_neutral()
                    canvas_after = painter.camera.get_canvas() if not painter.opt.simulate else full_sim_canvas
                    
                    real_canvases.append(canvas_after)
                    sim_canvases.append(full_sim_canvas.copy())

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
                    if painter.opt.discrete:
                        painter.writer.add_image('images/sim_canvas', full_sim_canvas/255., global_it)
                    all_colors = save_colors(colors)
                    painter.writer.add_image('paint_colors/are', all_colors/255., global_it)
                    painter.writer.add_image('target/target_discrete', target/255., global_it)

                global_it += 1
                consecutive_paints += 1

                # if painter.opt.adaptive and k >= len(instructions):
                #     break # all done

                # if global_it % 100 == 0:
                #     # to_gif(all_canvases)
                #     to_video(real_canvases, fn='real_canvases.mp4')
                #     to_video(sim_canvases, fn='sim_canvases.mp4')
    painter.clean_paint_brush()
    painter.clean_paint_brush()
    to_video(real_canvases, fn='/home/frida/Videos/frida/real_canvases{}.mp4'.format(str(time.time())))
    to_video(sim_canvases, fn='/home/frida/Videos/frida/sim_canvases{}.mp4'.format(str(time.time())))



def paint_planner_diffvg(painter, target, colors, labels, how_often_to_get_paint=5):
    global global_it
    painter.to_neutral()
    canvas_after = painter.camera.get_canvas()
    real_canvases = [canvas_after]
    consecutive_paints = 0
    camera_capture_interval = 8
    curr_color = -1

    target = discretize_with_labels(colors, labels)

    for it in tqdm(range(1)): # how many times to go visit planning file
        canvas_before = canvas_after 

        # Save data that the python3 file needs
        save_for_python3_file(painter, colors, target, canvas_after.copy())

        # Plan the new strokes with SawyerPainter/scripts/plan_all_strokes.py
        exit_code = subprocess.call(['python3.7', '/home/frida/ros_ws/src/intera_sdk/SawyerPainter/scripts/plan_all_strokes_diffvg.py']+sys.argv[1:]+['--global_it', str(global_it)])
        if exit_code != 0:
            print('exit code', exit_code)
            return
        show_img(canvas_before/255., title="Ready to start painting. Ensure mixed paint is provided and then exit this to start painting.")

        # Run Planned Strokes
        instructions = pickle.load(open(os.path.join(painter.opt.cache_dir, "next_brush_strokes_diffvg.pkl"),'r'))
        
        for instruction in tqdm(instructions):
            canvas_before = canvas_after
            xs, ys, z, color = instruction['xs'], 1-instruction['ys'], instruction['z'], instruction['color']
            color *= 255.
            color_ind, color_discrete = nearest_color(color, colors)

            # Clean paint brush and/or get more paint
            new_paint_color = color_ind != curr_color
            if new_paint_color:
                dark_to_light = np.mean(colors[curr_color]) < np.mean(colors[color_ind])
                # if dark_to_light and curr_color != -1:
                #     painter.clean_paint_brush() # Really clean this thing
                #     painter.clean_paint_brush()
                #     show_img(target/255., title="About to start painting with a lighter color")
                # painter.clean_paint_brush()
                # painter.clean_paint_brush()
                curr_color = color_ind
            if consecutive_paints >= how_often_to_get_paint or new_paint_color:
                painter.get_paint(color_ind)
                consecutive_paints = 0

            # Paint the brush stroke
            paint_diffvg(painter, xs, ys, z)

            if global_it%camera_capture_interval == 0:
                painter.to_neutral()
                canvas_after = painter.camera.get_canvas()
                
                real_canvases.append(canvas_after)


            if global_it%5==0: # loggin be sloggin
                if not painter.opt.simulate:
                    painter.writer.add_image('images/canvas', canvas_after/255., global_it)
                all_colors = save_colors(colors)
                painter.writer.add_image('paint_colors/are', all_colors/255., global_it)
                painter.writer.add_image('target/target_discrete', target/255., global_it)

            global_it += 1
            consecutive_paints += 1

    to_video(real_canvases, fn='real_canvases.mp4')




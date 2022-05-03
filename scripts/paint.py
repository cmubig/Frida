#! /usr/bin/env python
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import os
import time
import sys
import cv2
import numpy as np
import argparse
import datetime

from paint_utils import *
from painter import Painter
from options import Options
from paint_planner import paint_finely

import torch
import lpips

from tensorboard import TensorBoard
date_and_time = datetime.datetime.now()
run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
writer = TensorBoard('painting/{}'.format(run_name))


if __name__ == '__main__':
    
    opt = Options()
    opt.gather_options()

    painter = Painter(opt, robot=None if opt.simulate else "sawyer", 
        use_cache=opt.use_cache, writer=writer)

    if not opt.simulate: painter.robot.display_frida()

    target = load_img(opt.target)
    
    canvas = painter.camera.get_canvas()
    target = cv2.resize(target, (canvas.shape[1], canvas.shape[0]))
    target = np.array(target)
    writer.add_image('target/real', target, 0)
    full_sim_canvas = canvas.copy()

    colors = get_colors(cv2.resize(target, (512, 512)), n_colors=opt.n_colors)
    colors = sorted(colors, key=lambda l:np.mean(l), reverse=True) # Light to dark
    all_colors = save_colors(colors)
    writer.add_image('paint_colors/should_be', all_colors/255., 0)


    # colors = get_mixed_paint_colors(painter.camera.get_color_correct_image(), opt.n_colors, use_cache=opt.use_cache)
    # all_colors = save_colors(colors)
    # show_img(all_colors/255.)
    # writer.add_image('paint_colors/actual', all_colors/255., 0)

    # Change target image to only use actual paint colors
    target_not_discrete = target.copy()
    # target = discretize_image(target, colors)
    # writer.add_image('target/discrete', target/255., 0)

    # Ensure that x,y on the canvas photograph is x,y for the robot interacting with the canvas
    painter.coordinate_calibration(use_cache=opt.use_cache)

    # Use simulated painting as target
    # target = paint_in_simulation(target_not_discrete, canvas, painter, colors)
    # writer.add_image('target/simulated', target, 0)
    # show_img(target/255., title="Simulated painting. Close this popup to start painting this.")
    

    # show_img(all_colors/255., title="Mix these colors, then exit this popup to start painting")

    # paint_coarsely(painter, target, colors)
    paint_finely(painter, target_not_discrete, colors)

    painter.to_neutral()

    painter.robot.good_night_robot()




    # painter.robot.take_picture()

    # instructions = load_instructions(args.file)

    # curr_color = -1
    # since_got_paint = 0
    
    # for instr in tqdm(instructions[args.continue_ind:]):
    #     if args.type == 'cubic_bezier':
    #         # Cubic Bezier
    #         path = instr[2:]
    #         path = np.reshape(path, (len(path)//2, 2))
    #         color = instr[1]
    #         radius = instr[0]
    #         if color != curr_color:
    #             painter.clean_paint_brush()

    #         if color != curr_color or since_got_paint == painter.GET_PAINT_FREQ:
    #             painter.get_paint(color)
    #             since_got_paint = 0
    #         since_got_paint += 1

    #         painter.paint_cubic_bezier(path)
    #         curr_color = color
    #     else:
    #         # Quadratic Bezier Curve
    #         p0, p1, p2 = instr[0:2], instr[2:4], instr[4:6]
    #         color = instr[12]
    #         radius = instr[6]
    #         if color != curr_color:
    #             painter.clean_paint_brush()

    #         if color != curr_color or since_got_paint == painter.GET_PAINT_FREQ:
    #             painter.get_paint(color)
    #             since_got_paint = 0
    #         since_got_paint += 1

    #         painter.paint_quadratic_bezier(p0, p1, p2)
    #         curr_color = color
    #     # take a picture
    # painter.clean_paint_brush()

    # # for i in range(12):
    # #     get_paint(i)
    # # get_paint(0)
    # # get_paint(5)
    # # get_paint(6)
    # # get_paint(11)

    # # clean_paint_brush()

    # # paint_bezier_curve((0,0),(0,0),(0,0))
    # # paint_bezier_curve((0,1),(0,1),(0,1)) # top-left
    # # paint_bezier_curve((1,0),(1,0),(1,0)) # bottom-right
    # # paint_bezier_curve((1,1),(1,1),(1,1))
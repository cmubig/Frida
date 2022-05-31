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
from paint_planner import paint_finely, paint_planner_new

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

    # painter.camera.canvas = target.copy() + np.random.randn(*target.shape)*0.0001


    colors = get_colors(cv2.resize(target, (256, 256)), n_colors=opt.n_colors)
    colors = sorted(colors, key=lambda l:np.mean(l), reverse=True) # Light to dark
    all_colors = save_colors(colors)
    writer.add_image('paint_colors/should_be', all_colors/255., 0)



    # colors = get_mixed_paint_colors(painter.camera.get_color_correct_image(), opt.n_colors, use_cache=opt.use_cache)
    # all_colors = save_colors(colors)
    # show_img(all_colors/255.)
    # writer.add_image('paint_colors/actual', all_colors/255., 0)

    # Change target image to only use actual paint colors
    # target_discrete = discretize_image(target, colors)
    # writer.add_image('target/discrete', target_discrete/255., 0)
    # target_discrete = discretize_image_old(target, colors)
    # # from scipy.signal import medfilt
    # # target_discrete = medfilt(cv2.resize(target_discrete, (256,256)), kernel_size=5)
    # # target_discrete = cv2.resize(target_discrete, (canvas.shape[1], canvas.shape[0]))
    # writer.add_image('target/discrete_l2', target_discrete/255., 0)

    # Use simulated painting as target
    # target = paint_in_simulation(target_not_discrete, canvas, painter, colors)
    # writer.add_image('target/simulated', target, 0)
    # show_img(target/255., title="Simulated painting. Close this popup to start painting this.")
    

    if not opt.simulate:
        show_img(all_colors/255., title="Mix these colors, then exit this popup to start painting")


    paint_planner_new(painter, target, colors)

    # paint_coarsely(painter, target, colors)
    # paint_finely(painter, target_discrete, colors)

    painter.to_neutral()

    painter.robot.good_night_robot()



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
from paint_planner import paint_planner_new

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

    # painter.paint_continuous_stroke_library()

    target = load_img(opt.target)

    canvas = painter.camera.get_canvas()
    target = cv2.resize(target, (canvas.shape[1], canvas.shape[0]))
    target = np.array(target)
    writer.add_image('target/real', target, 0)


    colors, labels = get_colors(cv2.resize(target, (256, 256)), n_colors=opt.n_colors)
    all_colors = save_colors(colors)
    writer.add_image('paint_colors/should_be', all_colors/255., 0)
    # with open(os.path.join(painter.opt.cache_dir, 'color_labels.npy'), 'wb') as f:
    #     np.save(f, labels)

    if not opt.simulate:
        show_img(all_colors/255., title="Mix these colors, then exit this popup to start painting")

    # if opt.simulate:
    #     colors += np.random.randn(*colors.shape)*10
    colors = np.clip(colors, a_min=20, a_max=220)
    paint_planner_new(painter, target, colors, labels)


    painter.to_neutral()

    painter.robot.good_night_robot()



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

from painter import Painter
from options import Options
from paint_planner import paint_planner_new#, paint_planner_diffvg

from my_tensorboard import TensorBoard


if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    writer.add_text('args', str(sys.argv), 0)

    painter = Painter(opt, robot=None if opt.simulate else "sawyer", 
        use_cache=opt.use_cache, writer=writer)

    

    if not opt.simulate: painter.robot.display_frida()

    # painter.paint_continuous_stroke_library()

    # target = load_img(opt.target)

    canvas = painter.camera.get_canvas()
    # target = cv2.resize(target, (canvas.shape[1], canvas.shape[0]))
    # target = np.array(target)
    # writer.add_image('target/real', target, 0)


    # if opt.use_colors_from is not None:
    #     color_img = load_img(opt.use_colors_from)
    #     colors, labels = get_colors(cv2.resize(color_img, (128, 128)), n_colors=opt.n_colors)
    #     all_colors = save_colors(colors)
    #     writer.add_image('paint_colors/should_be', all_colors/255., 0)

    if not opt.simulate:
        # show_img(all_colors/255., title="Mix these colors, then exit this popup to start painting")
        try:
            input('Make sure blank canvas is exposed. Press enter when you are ready for the paint planning to start. Use tensorboard to see which colors to paint.')
        except SyntaxError:
            pass

    # if opt.simulate:
    #     colors += np.random.randn(*colors.shape)*10
    # colors = np.clip(colors, a_min=20, a_max=220)
    # if opt.diffvg:
    #     paint_planner_diffvg(painter)
    # else:
    paint_planner_new(painter)


    painter.to_neutral()

    painter.robot.good_night_robot()



#! /usr/bin/env python
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import sys
import cv2
import datetime
import numpy as np

import torch
from paint_utils3 import canvas_to_global_coordinates, get_colors, nearest_color, random_init_painting, save_colors, show_img

from painter import Painter
from options import Options
# from paint_planner import paint_planner_new

from my_tensorboard import TensorBoard
from painting_optimization import load_objectives_data, optimize_painting

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    painter = Painter(opt)
    opt = painter.opt # This necessary?

    if not opt.simulate:
        try:
            input('Make sure blank canvas is exposed. Press enter when you are ready for the paint planning to start. Use tensorboard to see which colors to paint.')
        except SyntaxError:
            pass

    # paint_planner_new(painter)

    painter.to_neutral()

    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render

    consecutive_paints = 0
    consecutive_strokes_no_clean = 0
    curr_color = -1

    color_palette = None
    if opt.use_colors_from is not None:
        color_palette = get_colors(cv2.resize(cv2.imread(opt.use_colors_from)[:,:,::-1], (256, 256)), 
                n_colors=opt.n_colors)
        opt.writer.add_image('paint_colors/using_colors_from_input', save_colors(color_palette), 0)

    current_canvas = painter.camera.get_canvas_tensor(h=h_render,w=w_render).to(device) / 255.

    load_objectives_data(opt)

    painting = random_init_painting(opt, current_canvas, opt.num_strokes, ink=opt.ink)
    painting.to(device)

    # Do the initial optimization
    painting, color_palette = optimize_painting(opt, painting, 
                optim_iter=opt.init_optim_iter, color_palette=color_palette)
    

    if not painter.opt.simulate:
        show_img(painter.camera.get_canvas()/255., 
                 title="Initial plan complete. Ready to start painting. \
                    Ensure mixed paint is provided and then exit this to \
                    start painting.")


    strokes_per_adaptation = int(len(painting) / opt.num_adaptations)
    # for adaptation_it in range(opt.num_adaptations):
    while len(painting) > 0:
        ################################
        ### Execute some of the plan ###
        ################################
        for stroke_ind in range(min(len(painting),strokes_per_adaptation)):
            stroke = painting.pop()            
            
            # Clean paint brush and/or get more paint
            if not painter.opt.ink:
                color_ind, _ = nearest_color(stroke.color_transform.detach().cpu().numpy(), 
                                             color_palette.detach().cpu().numpy())
                new_paint_color = color_ind != curr_color
                if new_paint_color or consecutive_strokes_no_clean > 12:
                    painter.clean_paint_brush()
                    painter.clean_paint_brush()
                    consecutive_strokes_no_clean = 0
                    curr_color = color_ind
                    new_paint_color = True
                if consecutive_paints >= opt.how_often_to_get_paint or new_paint_color:
                    painter.get_paint(color_ind)
                    consecutive_paints = 0

            # Convert the canvas proportion coordinates to meters from robot
            x, y = stroke.transformation.xt.item()*0.5+0.5, stroke.transformation.yt.item()*0.5+0.5
            x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
            x_glob, y_glob,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

            # Runnit
            stroke.execute(painter, x_glob, y_glob, stroke.transformation.a.item())

        #######################
        ### Update the plan ###
        #######################
        painter.to_neutral()
        current_canvas = painter.camera.get_canvas_tensor(h=h_render,w=w_render).to(device) / 255.
        painting.background_img = current_canvas
        painting, _ = optimize_painting(opt, painting, 
                    optim_iter=opt.optim_iter, color_palette=color_palette)



    # to_video(plans, fn=os.path.join(opt.plan_gif_dir,'sim_canvases{}.mp4'.format(str(time.time()))))
    # with torch.no_grad():
    #     save_image(painting(h*4,w*4, use_alpha=False), os.path.join(opt.plan_gif_dir, 'init_painting_plan{}.png'.format(str(time.time()))))


    painter.robot.good_night_robot()



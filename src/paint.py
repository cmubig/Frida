#! /usr/bin/env python
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import os
import sys
import time
import cv2
import datetime
import numpy as np
import pickle

import torch
from tqdm import tqdm
from paint_utils3 import canvas_to_global_coordinates, discretize_colors, get_colors, nearest_color, random_init_painting, save_colors, show_img, to_video

from painter import Painter
from options import Options

from my_tensorboard import TensorBoard
from painting_optimization import load_objectives_data, optimize_painting
from camera.webcam_portrait import WebcamInterface
import threading
import traceback
import matplotlib
matplotlib.use('TkAgg')

import logging
# disable warning messages from QObject::moveToThread: Current thread
logging.getLogger().setLevel(logging.CRITICAL)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PaintingExecutor():
    def __init__(self):

        self.opt = Options()
        self.opt.gather_options()

        date_and_time = datetime.datetime.now()
        run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
        self.opt.writer = TensorBoard('{}/{}'.format(self.opt.tensorboard_dir, run_name))
        self.opt.writer.add_text('args', str(sys.argv), 0)

        self.painter = Painter(self.opt)
        self.opt = self.painter.opt # This necessary?
        
        self.consecutive_paints = 0
        self.consecutive_strokes_no_clean = 0
        self.curr_color = -1

        self.done_painting = True
        self.new_plan_ready = False
        
        self.w_render = int(self.opt.render_height * (self.opt.CANVAS_WIDTH_M/self.opt.CANVAS_HEIGHT_M))
        self.h_render = int(self.opt.render_height)
        self.initial_canvas = self.painter.camera.get_canvas_tensor(h=self.h_render,w=self.w_render).to(device) / 255.
        self.initial_canvas[:,:,:,:] = 1.0

        self.lock = threading.Lock()
        # create execution thread for running the function execute
        self.execution_thread = threading.Thread(target=self.execution_thread)
        self.execution_thread.start()
        print('Starting webcam interface')
        self.webcam_interface = WebcamInterface(self.opt.init_optim_iter, self.opt.num_strokes, ratio= (self.opt.CANVAS_WIDTH_M/self.opt.CANVAS_HEIGHT_M))


    def compute_plan(self):

        if self.opt.webcam_interface:
            #set --objective_data to the path of the webcam portrait
            print(self.opt.init_optim_iter, type(self.opt.init_optim_iter))
            print(f'last portrait: {self.webcam_interface.last_portrait_path}')
            self.opt.objective_data = [self.webcam_interface.last_portrait_path]
            # print(type(opt.objective_data), opt.objective_data)
            error, num_iter, num_strokes = self.webcam_interface.run()
            print(num_iter, type(num_iter))
            self.opt.init_optim_iter = num_iter
            self.opt.num_strokes = num_strokes
            # if error:
            #     print('exiting')
            #     sys.exit(1)


        # if not self.opt.simulate:
        #     try:
        #         input('Make sure blank canvas is exposed. Press enter when you are ready for the paint planning to start. Use tensorboard to see which colors to paint.')
        #     except SyntaxError:
        #         pass


        self.opt.w_render, self.opt.h_render = self.w_render, self.h_render

        if self.opt.use_colors_from is not None:
            self.color_palette_plan = get_colors(cv2.resize(cv2.imread(self.opt.use_colors_from)[:,:,::-1], (256, 256)), 
                    n_colors=self.opt.n_colors).to(device)
            print(self.color_palette_plan)
            self.opt.writer.add_image('paint_colors/using_colors_from_input', save_colors(self.color_palette_plan), 0)

        self.lock.acquire()
        current_canvas = self.initial_canvas
        self.lock.release()

        load_objectives_data(self.opt)

        self.painting_plan = random_init_painting(self.opt, current_canvas, self.opt.num_strokes, ink=self.opt.ink)
        self.painting_plan.to(device)

        with torch.no_grad():
            for bs in self.painting_plan.brush_strokes:
                bs.color_transform *= 0

        if self.opt.use_colors_from is not None:
            discretize_colors(self.painting_plan, self.color_palette_plan)
        print(self.opt.recover_painting)
        if self.opt.recover_painting:
            self.recover_painting()
            self.opt.recover_painting = False
        else:
            # Do the initial optimization
            self.painting_plan, self.color_palette_plan = optimize_painting(self.opt, self.painting_plan, 
                        optim_iter=self.opt.init_optim_iter, color_palette=self.color_palette_plan)
            
        if self.opt.save_painting:
            # create directory to save painting
            if not os.path.exists(self.opt.painting_path):
                os.makedirs(self.opt.painting_path)

            #save painting as pickle object
            with open(self.opt.painting_path + '/painting.pkl', 'wb') as f:
                pickle.dump(self.painting_plan, f)
            
            # save color_palette
            with open(self.opt.painting_path + '/color_palette.pkl', 'wb') as f:
                pickle.dump(self.color_palette_plan, f)

        self.new_plan_ready = True

    def recover_painting(self):

        print(" ------- Recovering painting from last run -------")
        #load painting
        painting_name = self.opt.painting_name if len(self.opt.painting_name) > 0 else "painting" 
        painting_file = self.opt.painting_path + '/' + painting_name +'.pkl'
        print(painting_file)
        try:
            with open(painting_file, 'rb') as f:
                self.painting_plan = pickle.load(f)
            print("Loaded!!")
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print(f"couldn't load the painting plan: {painting_file}")

        with open(self.opt.painting_path + '/color_palette.pkl', 'rb') as f:
            self.color_palette_plan = pickle.load(f)

        self.new_plan_ready = True

    def interface_loop(self):
        while True:
            try:
                print("starting interface loop")
                self.compute_plan()
            except Exception as e:
                print(traceback.format_exc())

                print(e)
            time.sleep(1)

    def execution_thread(self):
        while True:
            if self.done_painting and self.new_plan_ready:
                self.done_painting = False
                self.new_plan_ready = False
                print('================== Executing new plan ================')
                self.execute_plan()
                print('================== Done executing plan ================')
                self.done_painting = True

            time.sleep(1)

    def execute_plan(self):
        
        self.painting = self.painting_plan
        self.color_palette = self.color_palette_plan
        
        self.painter.to_neutral()
        # painting.param2img.renderer.set_render_size(size_x=w_render, size_y=h_render)
        # painting.param2img.renderer.set_render_size(size_x=h_render, size_y=h_render)
        
        # Log colors so user can start to mix them
        if not self.opt.ink:
            self.opt.writer.add_image('paint_colors/mix_these_colors', save_colors(self.color_palette), 0)
            self.opt.writer.add_image('paint_colors/mix_these_colors', save_colors(self.color_palette), 1)


        if not self.opt.simulate and not self.opt.ink:
            big_img = save_colors(self.color_palette)
            colors_title="Initial plan complete. Ready to start painting. Ensure mixed paint is provided and then exit this to start painting."
            # show image using cv2 
            # cv2.imshow("colors", big_img)
            # print(colors_title)
            # # add title to the image
            # cv2.setWindowTitle("colors", colors_title)
            # keyCode = cv2.waitKey(50)
            # while cv2.getWindowProperty('colors', 0) >= 0 and keyCode != 27:
            #     keyCode = cv2.waitKey(50)
            done = False
            while not done:
                try:
                    show_img(big_img, 
                            title=colors_title)
                    done = True
                except Exception as e:
                    print(e)
                    # input('Press enter to continue')
                time.sleep(1)


        strokes_per_adaptation = int(len(self.painting) / self.opt.num_adaptations)
        strokes_executed, canvas_photos = 0, []
        # for adaptation_it in range(self.opt.num_adaptations):
        while len(self.painting) > 0:
            ################################
            ### Execute some of the plan ###
            ################################
            for stroke_ind in tqdm(range(min(len(self.painting),strokes_per_adaptation))):
                stroke = self.painting.pop()            
                
                # Clean paint brush and/or get more paint
                if not self.painter.opt.ink:
                    color_ind, _ = nearest_color(stroke.color_transform.detach().cpu().numpy(), 
                                                self.color_palette.detach().cpu().numpy())
                    new_paint_color = color_ind != self.curr_color
                    if new_paint_color or self.consecutive_strokes_no_clean > 12:
                        if new_paint_color: self.painter.clean_paint_brush()
                        # self.painter.clean_paint_brush()
                        self.consecutive_strokes_no_clean = 0
                        self.curr_color = color_ind
                        new_paint_color = True
                    if self.consecutive_paints >= self.opt.how_often_to_get_paint or new_paint_color:
                        self.painter.get_paint(color_ind)
                        self.consecutive_paints = 0

                # Convert the canvas proportion coordinates to meters from robot
                x, y = stroke.xt.item(), stroke.yt.item()
                y = 1-y
                x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
                x_glob, y_glob,_ = canvas_to_global_coordinates(x,y,None,self.painter.opt)

                # Runnit
                start_t = time.time()
                stroke.execute(self.painter, x_glob, y_glob, stroke.a.item(), fast=True)
                end_t = time.time()
                if end_t - start_t < 0.01:
                    print("CONTROLER ERROR")

                strokes_executed += 1
                self.consecutive_paints += 1
                self.consecutive_strokes_no_clean += 1

                # if strokes_executed % self.opt.log_photo_frequency == 0:
                #     self.painter.to_neutral()
                #     canvas_photos.append(self.painter.camera.get_canvas())
                #     self.painter.writer.add_image('images/canvas', canvas_photos[-1]/255., strokes_executed)

            #######################
            ### Update the plan ###
            #######################
            self.painter.to_neutral()
            self.lock.acquire()
            current_canvas = self.painter.camera.get_canvas_tensor(h=self.h_render,w=self.w_render).to(device) / 255.
            self.lock.release()
            self.painting.background_img = current_canvas
            self.painting, _ = optimize_painting(self.opt, self.painting, 
                        optim_iter=self.opt.optim_iter, color_palette=self.color_palette)

        self.painter.to_neutral()
        canvas_photos.append(self.painter.camera.get_canvas())
        self.painter.writer.add_image('images/canvas', canvas_photos[-1]/255., strokes_executed)

        if not self.painter.opt.ink:
            for i in range(1): self.painter.clean_paint_brush()

        to_video(canvas_photos, fn=os.path.join(self.opt.plan_gif_dir,'sim_canvases{}.mp4'.format(str(time.time()))))
        # with torch.no_grad():
        #     save_image(painting(h*4,w*4, use_alpha=False), os.path.join(opt.plan_gif_dir, 'init_painting_plan{}.png'.format(str(time.time()))))
        # save_image(canvas_photos[-1], os.path.join(opt.plan_gif_dir, 'init_painting_plan{}.png'.format(str(time.time()))))





if __name__ == '__main__':
    painter_executor = PaintingExecutor()
    painter_executor.interface_loop()
    del painter_executor
    exit(0)




##########################################################
#################### Copyright 2023 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################


import datetime
import sys
import time
import torch
from torchvision.transforms import Resize
from tqdm import tqdm
from pynput import keyboard

from paint_utils3 import canvas_to_global_coordinates, format_img

from painter import Painter
from options import Options
from my_tensorboard import TensorBoard

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def flip_img(img):
    return torch.flip(img, dims=(2,3))

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    painter = Painter(opt)
    opt = painter.opt 

    painter.to_neutral()

    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render

    consecutive_paints = 0
    consecutive_strokes_no_clean = 0
    curr_color = -1

    painting = torch.load(opt.saved_plan)

    print('Press any key to pause/continue')

    is_paused = False
    def on_press(key):
        try:
            global is_paused
            is_paused = not is_paused
            # print('alphanumeric key {0} pressed'.format(
            #     key.char))
            if is_paused:
                print("Paused")
            else:
                print('Resuming')
        except AttributeError:
            # print('special key {0} pressed'.format(
            #     key))
            print('some error')
    
    # ...or, in a non-blocking fashion:
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    # Execute plan
    for stroke_ind in tqdm(range(len(painting)), desc="Executing plan"):
        while is_paused:
            time.sleep(0.1)

        stroke = painting.pop()

        # Convert the canvas proportion coordinates to meters from robot
        x, y = stroke.xt.item(), stroke.yt.item()
        y = 1-y
        x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
        x_glob, y_glob,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

        # Runnit
        stroke.execute(painter, x_glob, y_glob, stroke.a.item(), fast=True)

        if opt.simulate:
            time.sleep(1)

    painter.to_neutral()

    current_canvas = painter.camera.get_canvas_tensor() / 255.
    current_canvas = flip_img(current_canvas)
    opt.writer.add_image('images/{}_4_canvas_after_drawing'.format(0), format_img(current_canvas), 0)
    current_canvas = Resize((h_render, w_render), antialias=True)(current_canvas)
    
    if not painter.opt.ink:
        painter.clean_paint_brush()
        painter.clean_paint_brush()
    
    painter.to_neutral()

    painter.robot.good_night_robot()

##########################################################
#################### Copyright 2023 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################


import datetime
import random
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import easygui
from torchvision.transforms import Resize
from tqdm import tqdm
from PIL import Image

from cofrida import get_instruct_pix2pix_model
from paint_utils3 import canvas_to_global_coordinates, format_img, get_colors, initialize_painting, nearest_color, random_init_painting, save_colors, show_img
from painting_optimization import optimize_painting

from painter import Painter
from options import Options
from my_tensorboard import TensorBoard
from train_transformer_cofrida_strokes import StrokePredictor

# For Audio Recording and Transcription 
# from audio_test import get_text_from_audio

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def get_cofrida_image_to_draw(cofrida_model, curr_canvas_pil):
    ''' Create a GUI to allow the user to see different CoFRIDA options and pick
    '''

    text_prompt = easygui.enterbox("What do you want the robot to draw?")

    while True:
        with torch.no_grad():
            target_img = cofrida_model(
                text_prompt, 
                curr_canvas_pil, 
                num_inference_steps=20, 
                # generator=generator,
                num_images_per_prompt=1,
                # image_guidance_scale=2.5,#1.5 is default
                # image_guidance_scale = 1.5 if op == 0 else random.uniform(1.01, 2.5)
            ).images[0]
        target_img.save('target_img_temp.png')
        image = 'target_img_temp.png'
        msg = "Should I draw this?"
        choices = ["Yes","No, give another option","No, I want to type a new text description"]
        yes = choices[0]
        new_option = choices[1]
        new_prompt = choices[2]
        reply = easygui.buttonbox(msg, image=image, choices=choices)
        
        if reply == yes:
            break 
        elif reply == new_option:
            continue
        elif reply == new_prompt:
            text_prompt = easygui.enterbox("What do you want the robot to draw?")
            continue

        
 
    # plt.imshow(target_img)
    # plt.show()
    target_img = torch.from_numpy(np.array(target_img)).permute(2,0,1).float().to(device)[None] / 255.
    # target_img = Resize((h,w), antialias=True)(target_img)
    return text_prompt, target_img

def flip_img(img):
    return torch.flip(img, dims=(2,3))

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()


    cofrida_model = get_instruct_pix2pix_model(
                lora_weights_path=opt.cofrida_model, 
                device=device)
    cofrida_model.set_progress_bar_config(disable=True)

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

    stroke_predictor = None 
    if opt.continue_training is not None:
        stroke_predictor = StrokePredictor(opt, 
            n_strokes=opt.n_predicted_strokes)
        stroke_predictor.load_state_dict(torch.load(opt.continue_training))
        stroke_predictor.to(device)
        stroke_predictor.eval()

    consecutive_paints = 0
    consecutive_strokes_no_clean = 0
    curr_color = -1

    color_palette = None
    if opt.use_colors_from is not None:
        color_palette = get_colors(cv2.resize(cv2.imread(opt.use_colors_from)[:,:,::-1], (256, 256)), 
                n_colors=opt.n_colors)
        opt.writer.add_image('paint_colors/using_colors_from_input', save_colors(color_palette), 0)

    for i in range(9): # Max number of turns to take

        ##################################
        ########## Hooman Turn ###########
        ##################################
        
        current_canvas = painter.camera.get_canvas_tensor() / 255.
        current_canvas = flip_img(current_canvas)
        opt.writer.add_image('images/{}_0_canvas_start'.format(i), format_img(current_canvas), 0)
        current_canvas = Resize((h_render,w_render), antialias=True)(current_canvas)

        # try:
        #     input('\nFeel free to draw, then press enter when done.')
        # except SyntaxError:
        #     pass
        msg = "Feel free to draw if you'd like"
        choices = ["Done"]
        reply = easygui.buttonbox(msg, choices=choices)

        current_canvas = painter.camera.get_canvas_tensor() / 255.
        current_canvas = flip_img(current_canvas)
        opt.writer.add_image('images/{}_1_canvas_after_human'.format(i), format_img(current_canvas), 0)
        current_canvas = Resize((h_render,w_render), antialias=True)(current_canvas)

        #################################
        ########## Robot Turn ###########
        #################################

        curr_canvas = painter.camera.get_canvas()
        curr_canvas = np.flip(curr_canvas, axis=(0,1))
        # dark_inds = curr_canvas.mean(axis=2) < 0.81*255
        # curr_canvas[dark_inds] = 5
        curr_canvas_pil = Image.fromarray(curr_canvas.astype(np.uint8)).resize((512,512))
        curr_canvas_pil = curr_canvas_pil#.convert("L").convert('RGB')

        # Let the user generate and choose cofrida images to draw
        text_prompt, target_img = get_cofrida_image_to_draw(cofrida_model, 
                                                            curr_canvas_pil)
        opt.writer.add_image('images/{}_2_target_from_cofrida_{}'.format(i, text_prompt), format_img(target_img), 0)
        target_img = Resize((h_render, w_render), antialias=True)(target_img)
        target_img = flip_img(target_img) # Should be upside down for planning

        # Ask for how many strokes to use
        # num_strokes = 90#int(input("How many strokes to use in this plan?\n:"))
        # num_strokes = int(input("How many strokes to use in this plan?\n:"))
        num_strokes = 20

        
        # Generate initial (random plan)
        # painting = random_init_painting(opt, current_canvas.to(device), num_strokes, ink=opt.ink).to(device)
        current_canvas = flip_img(current_canvas) # Should look upside down / real
        painting = initialize_painting(opt, num_strokes, target_img, 
                                       current_canvas.to(device), opt.ink, 
                                       stroke_predictor=stroke_predictor,
                                       device=device)
        color_palette = None #TODO: support input fixed palette

        # Set variables for planning algorithm
        opt.objective = ['clip_conv_loss']
        opt.objective_data_loaded = [target_img]
        opt.objective_weight = [1.0]
        
        # Get the plan
        painting, _ = optimize_painting(opt, painting, 
                    optim_iter=opt.optim_iter, color_palette=color_palette,
                    log_title='{}_3_plan'.format(i))
        
        # Warn the user plan is ready. Get paint ready.
        # if not painter.opt.simulate:
        #     show_img(painter.camera.get_canvas()/255., 
        #             title="Initial plan complete. Ready to start painting."
        #                 + "Ensure mixed paint is provided and then exit this to "
        #                 + "start painting.")

        # Execute plan
        for stroke_ind in tqdm(range(len(painting)), desc="Executing plan"):
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
            x, y = stroke.xt.item(), stroke.yt.item()
            y = 1-y
            x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
            x_glob, y_glob,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

            # Runnit
            stroke.execute(painter, x_glob, y_glob, stroke.a.item(), fast=True)

        painter.to_neutral()

        current_canvas = painter.camera.get_canvas_tensor() / 255.
        current_canvas = flip_img(current_canvas)
        opt.writer.add_image('images/{}_4_canvas_after_drawing'.format(i), format_img(current_canvas), 0)
        current_canvas = Resize((h_render, w_render), antialias=True)(current_canvas)
    
    if not painter.opt.ink:
        painter.clean_paint_brush()
        painter.clean_paint_brush()
    
    painter.to_neutral()

    painter.robot.good_night_robot()
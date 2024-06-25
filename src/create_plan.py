
##########################################################
#################### Copyright 2023 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################


import datetime
import os
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
from paint_utils3 import canvas_to_global_coordinates, format_img, get_colors, initialize_painting, load_img, nearest_color, random_init_painting, save_colors, show_img
from painting_optimization import optimize_painting

from painter import Painter
from options import Options
from my_tensorboard import TensorBoard

# For Audio Recording and Transcription 
# from audio_test import get_text_from_audio

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def get_cofrida_image_to_draw(cofrida_model, curr_canvas_pil, n_ai_options):
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

def define_prompts_dictionary():
    # # Hand crafted prompts.
    prompts_dictionary = {} 

    prompts_dictionary['Painting0']['InitialPrompt'] = 'A black and white sharpie drawing of a single tree on a plain.'
    prompts_dictionary['Painting1']['InitialPrompt'] = 'A black and white sharpie drawing of a simple park with a bench.'
    prompts_dictionary['Painting2']['InitialPrompt'] = 'A black and white sharpie drawing of a rainy city street.'
    prompts_dictionary['Painting3']['InitialPrompt'] = 'A black and white sharpie drawing of a mountain silhouette.'
    prompts_dictionary['Painting4']['InitialPrompt'] = 'A black and white sharpie drawing of a small boat sailing on a plain lake.'
    prompts_dictionary['Painting5']['InitialPrompt'] = 'A black and white sharpie drawing of a ruins of an italian building.'
    prompts_dictionary['Painting6']['InitialPrompt'] = 'A black and white sharpie drawing of a desert with undulating sand dunes'
    prompts_dictionary['Painting7']['InitialPrompt'] = 'A black and white sharpie drawing of a twilight suburban street.'
    prompts_dictionary['Painting8']['InitialPrompt'] = 'A black and white sharpie drawing of a park trail with trees on either side.'
    prompts_dictionary['Painting9']['InitialPrompt'] = 'A black and white sharpie drawing of a simple grassy plain.'

    prompts_dictionary['Painting0']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a hilly landscape and a single tree on top of it.'
    prompts_dictionary['Painting1']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a quiet park with a single bench beneath a single tree.'
    prompts_dictionary['Painting2']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a rainy city street with puddles reflecting streetlights.'
    prompts_dictionary['Painting3']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a mountain range with sparse alpine trees.'
    prompts_dictionary['Painting4']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a boat sailing on a pretty lake with small waves'
    prompts_dictionary['Painting5']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a italian monument in Rome'
    prompts_dictionary['Painting6']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a undulating sand dunes with a few palm trees'
    prompts_dictionary['Painting7']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a twilight suburban street with a few buildings.'
    prompts_dictionary['Painting8']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a forest scene with a log cabin in the middle with lush trees in the background.'
    prompts_dictionary['Painting9']['MediumSubsequentPrompt'] = 'A black and white sharpie drawing of a grassy plain with one or two trees.'

    prompts_dictionary['Painting0']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a hilly landscape with a single majestic tree on top with clouds swirling in the sky.'
    prompts_dictionary['Painting1']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a serene city park with a quiet park bench beneath a canopy of several majestic trees.'
    prompts_dictionary['Painting2']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a rainy city street with puddles reflecting streetlights and umbrellas and a starry sky.'
    prompts_dictionary['Painting3']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a serene snow-capped mountain range with a lush forest of alpine trees on the base of the mountain'
    prompts_dictionary['Painting4']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a majestic ship sailing on waves on the ocean with a city skyline in the background.'
    prompts_dictionary['Painting5']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a majestic Collesium scene from Rome.'
    prompts_dictionary['Painting6']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a beautiful desert oasis with palm trees and a water body with sand dunes in the background.'
    prompts_dictionary['Painting7']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a twilight suburban street with a charming buildings and trees lining the street.'
    prompts_dictionary['Painting8']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a charming rustic cabin in a lush forested woodland.'
    prompts_dictionary['Painting9']['GoodSubsequentPrompt'] = 'A black and white sharpie drawing of a beautiful mountainous scene with lush fir trees.'

    return prompts_dictionary

def flip_img(img):
    return torch.flip(img, dims=(2,3))

def generate_image_and_plan(cofrida_model, opt, painting_prompt, prompt_key, current_canvas=None):
    
    #################################
    # Image generation phase. 
    #################################   

    # First get the prompt. 
    text_prompt = painting_prompt[prompt_key]
    # Get image_guidance_scale value. 
    image_guidance_dict = {'InitialPrompt': 1.9, 'MediumSubsequentPrompt': 1.2, 'GoodSubsequentPrompt': 1.}

    # Generate image. 
    target_img = cofrida_model(text_prompt, current_canvas, num_inference_steps=20, \
                               num_images_per_prompt=1, image_guidance_scale=image_guidance_dict[prompt_key]).images[0]
    
    # Log generated image. 
    opt.writer.add_image('images/target_from_cofrida_{0}_{1}'.format(prompt_key, text_prompt), format_img(target_img), 0)        
    target_img = Resize((h_render, w_render), antialias=True)(target_img)
    target_img = flip_img(target_img) # Should be upside down for planning

    #################################
    # Plan generation phase. 
    #################################

    # Ask for how many strokes to use
    # num_strokes = 90#int(input("How many strokes to use in this plan?\n:"))
    # num_strokes = int(input("How many strokes to use in this plan?\n:"))

    # Locking to 70. 
    num_strokes = 70
    
    # Generate initial (random plan)
    # painting = random_init_painting(opt, current_canvas.to(device), num_strokes, ink=opt.ink).to(device)
    curr_canvas_pt = flip_img(curr_canvas_pt) # Should look upside down / real
    painting = initialize_painting(opt, num_strokes, target_img, 
                                    curr_canvas_pt.to(device), opt.ink, device=device)
    color_palette = None #TODO: support input fixed palette

    # Set variables for planning algorithm
    opt.objective = ['clip_conv_loss']
    opt.objective_data_loaded = [target_img]
    opt.objective_weight = [1.0]
    
    # Get the plan
    painting, _ = optimize_painting(opt, painting, 
                optim_iter=opt.optim_iter, color_palette=color_palette,
                log_title='{}_3_plan'.format(0))
    
    rendered_painting, alphas = painting(opt.h_render, opt.w_render, 
                                         use_alpha=False, return_alphas=True)

    rendered_painting = flip_img(rendered_painting) # "Up side down" so that it looks right side up to viewer

    return painting, rendered_painting

def generate_all_plans(cofrida_model, opt, base_save_dir):

    # Generate the prompts dictionary. 
    prompts_dictionary = define_prompts_dictionary()

    if True: 

        # Temporarily set n_ai_options to 1 here to generate only 1 image per prompt. 
        n_ai_options = 1
        # Set some variables of how many plans etc. 
        n_paintings = 10

        # Create logger. 
        date_and_time = datetime.datetime.now()
        run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
        opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
        opt.writer.add_text('args', str(sys.argv), 0)

        # Create painter object. 
        painter = Painter(opt)
        opt = painter.opt 
        painter.to_neutral()

        # Set painting andimage size params. 
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
   
        # The current canvas image used here is really just the blank canvas image. 
        curr_canvas_pt = load_img(opt.background_image, h=h_render,w=w_render).to(device)/255.
        curr_canvas_np = curr_canvas_pt.detach().cpu().numpy()[0].transpose(1,2,0) * 255.
        # curr_canvas_np = np.flip(curr_canvas_np, axis=(0,1)) # Make it upaside down
        # dark_inds = curr_canvas.mean(axis=2) < 0.81*255
        # curr_canvas[dark_inds] = 5
        curr_canvas_pil = Image.fromarray(curr_canvas_np.astype(np.uint8)).resize((512,512))
        curr_canvas_pil = curr_canvas_pil#.convert("L").convert('RGB')

        # Blank Canvas variable. 
        blank_canvas_image = curr_canvas_pil
        

    # Iterate over tree for each painting. 
    for k in range(n_paintings):

        print("#####################")
        print("Running iteration ", k)
        print("#####################")

        ############################################################
        # Generate image / plan for Initial Prompt
        ############################################################
        
        # First generate image and plan from initial prompt. 
        painting_object, initial_prompt_target_image = generate_image_and_plan(cofrida_model, opt, prompt_dict['Painting{0}'.format(k)], prompt_key='InitialPrompt', blank_canvas_image)        
        
        # Set save directory. 
        save_dir_suffix = "Painting{0}/InitialPrompt".format(k)
        save_dir = os.path.join(base_save_dir, save_dir_suffix)
        
        # Save the image and the plan for this iteration. 
        save_image_and_plan(painting_object, initial_prompt_target_image, save_dir)
        
        ############################################################
        # Generate image / plan for Initial Prompt
        ############################################################
        
        # First generate image and plan from initial prompt. 
        painting_object, target_image = generate_image_and_plan(cofrida_model, opt, prompt_dict['Painting{0}'.format(k)], prompt_key='MediumSubsequentPrompt', initial_prompt_target_image)
        
        # Set save directory. 
        save_dir_suffix = "Painting{0}/MediumSubsequentPrompt".format(k)
        save_dir = os.path.join(base_save_dir, save_dir_suffix)
        
        # Save the image and the plan for this iteration. 
        save_image_and_plan(painting_object, target_image, save_dir)
        
        ############################################################
        # Generate image / plan for Medium Subsequent Prompt
        ############################################################
        
        # First generate image and plan from initial prompt. 
        painting_object, target_image = generate_image_and_plan(cofrida_model, opt, prompt_dict['Painting{0}'.format(k)], prompt_key='GoodSubsequentPrompt', initial_prompt_target_image)
        
        # Set save directory. 
        save_dir_suffix = "Painting{0}/GoodSubsequentPrompt".format(k)
        save_dir = os.path.join(base_save_dir, save_dir_suffix)
        
        # Save the image and the plan for this iteration. 
        save_image_and_plan(painting_object, opt, target_image, save_dir)
        


def save_image_and_plan(painting, opt, rendered_painting, save_dir):
    
    # Save plan and other stuff
    os.makedirs(save_dir, exist_ok=True)

    # Save rendering of the plan
    rendered_painting = flip_img(rendered_painting) # "Up side down" so that it looks right side up to viewer
    rendered_painting = rendered_painting.detach().cpu().numpy()[0].transpose(1,2,0) * 255.
    rendered_painting = Image.fromarray(rendered_painting.astype(np.uint8)).resize((512,512))
    rendered_painting.save(os.path.join(save_dir,'rendered_plan.png'))

    # # Save target image
    # target_img = flip_img(target_img) # "Up side down" so that it looks right side up to viewer
    # target_img = target_img.detach().cpu().numpy()[0].transpose(1,2,0) * 255.
    # target_img = Image.fromarray(target_img.astype(np.uint8)).resize((512,512))

    # Save plan
    torch.save(painting, os.path.join(save_dir, 'plan.pt'))

if __name__ == '__main__':

    # Create parameter and logger manager. 
    opt = Options()
    opt.gather_options()

    # Instantiate Co-FRIDA model. 
    cofrida_model = get_instruct_pix2pix_model(
                lora_weights_path=opt.cofrida_model, 
                device=device)
    cofrida_model.set_progress_bar_config(disable=True)

    save_dir = easygui.enterbox("What base directory should I save paintings and plans in ? (e.g., ./saved_plans/unique_name/)")

    # Process all prompts. 
    generate_all_plans(cofrida_model=cofrida_model, opt=opt, base_save_dir=save_dir)

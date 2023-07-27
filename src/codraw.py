
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################


import numpy as np
import torch
from torch import nn
import cv2
from tqdm import tqdm
import os
import time
import copy
# from paint_utils import save_colors

from options import Options
from painter import Painter
from strokes import simple_parameterization_to_real

# from torch_painting_models_continuous_concerted import *
from torch_painting_models_continuous import *
from style_loss import compute_style_loss
from sketch_loss.sketch_loss import compute_sketch_loss, compute_canny_loss
from audio_loss.audio_loss import compute_audio_loss, load_audio_file
from emotion_loss.emotion_loss import emotion_loss
from face.face_loss import face_loss, parse_face_data
from stable_diffusion.stable_diffusion_loss2 import stable_diffusion_loss, encode_text_stable_diffusion
from speech2emotion.speech2emotion import speech2emotion, speech2text

from clip_loss import clip_conv_loss, clip_model, clip_text_loss, clip_model_16, clip_fc_loss
import clip
from clip_attn.clip_attn import get_attention
import kornia as K

from paint_utils import canvas_to_global_coordinates, to_video
from paint_utils3 import *
from torchvision.utils import save_image
from torchvision.transforms import Resize

from test_controlnet import pipeline as sd_interactive_pipeline
# from create_data_controlnet import image_text_similarity

# python3 codraw.py  --use_cache --cache_dir caches/cache_6_6_cvpr/ --dont_retrain_stroke_model --robot xarm --brush_length 0.2 --ink   --lr_multiplier 0.2 --num_strokes 40


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

# Utilities

writer = None
local_it = 0 
plans = []

def log_progress(painting, log_freq=5, force_log=False, title='plan'):
    global local_it, plans
    local_it +=1
    if (local_it %log_freq==0) or force_log:
        with torch.no_grad():
            #np_painting = painting(h,w, use_alpha=False).detach().cpu().numpy()[0].transpose(1,2,0)
            #opt.writer.add_image('images/{}'.format(title), np.clip(np_painting, a_min=0, a_max=1), local_it)
            p = painting(h,w, use_alpha=False)
            p = format_img(p)
            opt.writer.add_image('images/{}'.format(title), p, local_it)
            
            plans.append((p*255.).astype(np.uint8))



def plan_from_image(opt, target_img, current_canvas):
    global colors
    # painting = initialize_painting(opt.num_strokes, target_img, current_canvas, opt.ink)

    stroke_batch_size = opt.num_strokes#100#64
    iters_per_batch =  200

    # painting = initialize_painting(stroke_batch_size, target_img, current_canvas, opt.ink)
    painting = initialize_painting(1, target_img, current_canvas, opt.ink)
    painting.to(device)
    log_progress(painting, log_freq=opt.log_frequency, force_log=True)

    c = 0
    total_its = (opt.num_strokes/stroke_batch_size)*iters_per_batch
    for i in range(2):#(range(0, opt.num_strokes, stroke_batch_size)):#, desc="Initializing"):
        with torch.no_grad():
            p = painting(h,w)
        painting = add_strokes_to_painting(painting, p[:,:3], stroke_batch_size, target_img, current_canvas, opt.ink)
        optims = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)

        # Learning rate scheduling. Start low, middle high, end low
        og_lrs = [o.param_groups[0]['lr'] if o is not None else None for o in optims]

        for it in tqdm(range(iters_per_batch), desc="Optim. {} Strokes".format(len(painting.brush_strokes))):
            for o in optims: o.zero_grad() if o is not None else None
            

            lr_factor = (1 - 2*np.abs(it/iters_per_batch - 0.5)) + 0.05
            for i_o in range(len(optims)):
                if optims[i_o] is not None:
                    optims[i_o].param_groups[0]['lr'] = og_lrs[i_o]*lr_factor

            p, alphas = painting(h, w, use_alpha=True, return_alphas=True)

            t = c / total_its
            c+=1 
            
            loss = 0
            loss += clip_conv_loss(target_img, p[:,:3]) 

            loss.backward()

            for o in optims: o.step() if o is not None else None
            painting.validate()
            # painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            # for o in optims: o.param_groups[0]['lr'] = o.param_groups[0]['lr'] * 0.95 if o is not None else None
            if not opt.ink:
                painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            # make sure hidden strokes get some attention
            # painting = randomize_brush_stroke_order(painting)
            # log_progress(painting, title='int_optimization_', log_freq=opt.log_frequency)#, force_log=True)
            

            if (it % 10 == 0 and it > (0.5*iters_per_batch)) or it > 0.9*iters_per_batch:
                if opt.use_colors_from is None:
                    # Cluster the colors from the existing painting
                    if not opt.ink:
                        colors = painting.cluster_colors(opt.n_colors)
                if not opt.ink:
                    discretize_colors(painting, colors)

            log_progress(painting, log_freq=opt.log_frequency)#, force_log=True)


    # if opt.use_colors_from is None:
    #     colors = painting.cluster_colors(opt.n_colors)
    # with open(os.path.join(opt.cache_dir, 'colors_updated.npy'), 'wb') as f:
    #     np.save(f, (colors.detach().cpu().numpy()*255).astype(np.uint8))


    # discretize_colors(painting, colors)
    painting = sort_brush_strokes_by_location(painting, bin_size=opt.bin_size)
    log_progress(painting, force_log=True, log_freq=opt.log_frequency)

    return painting


if __name__ == '__main__':
    global opt
    opt = Options()
    opt.gather_options()

    # painting = calibrate(opt)

    opt.writer = create_tensorboard(log_dir=opt.tensorboard_dir)

    global h, w, colors, current_canvas, text_features, style_img, sketch
    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    w = int((opt.max_height/h)*w)
    h = int(opt.max_height)

    target_img = load_img('/home/frida/Downloads/id36_pass0_100strokes.jpg',h=h, w=w).to(device)/255.

    from robot import XArm, SimulatedRobot
    robot = SimulatedRobot() if opt.simulate else XArm(debug=True)
    robot.good_morning_robot()

    painter = Painter(opt, robot=None if opt.simulate else opt.robot, 
        use_cache=opt.use_cache, writer=opt.writer)
    
    corners = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]]).astype(np.float32)
    for i in range(len(corners)):
        corner = corners[i]
        x, y = corner[0], corner[1]
        x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
        x,y,_ = canvas_to_global_coordinates(x,y,None,painter.opt)
        if i == 0:
            painter.move_to(x,y,painter.Z_CANVAS+0.03)
        painter.move_to(x,y,painter.Z_CANVAS)
        if i == 4:
            painter.move_to(x,y,painter.Z_CANVAS+0.03)
    painter.to_neutral()
    
    for i in range(9):
        current_canvas = painter.camera.get_canvas()
        current_canvas = torch.from_numpy(current_canvas).permute(2,0,1).float().to(device)[None] / 255.
        current_canvas = Resize((h,w))(current_canvas)
        print('curr canva', current_canvas.shape)

        opt.writer.add_image('images/{}_0_canvas_start'.format(i), format_img(current_canvas), 0)


        try:
            input('\nFeel free to draw, then press enter when done.')
        except SyntaxError:
            pass


        current_canvas = painter.camera.get_canvas()
        current_canvas = torch.from_numpy(current_canvas).permute(2,0,1).float().to(device)[None] / 255.
        current_canvas = Resize((h,w))(current_canvas)
        opt.writer.add_image('images/{}_1_canvas_after_human'.format(i), format_img(current_canvas), 0)

        #text_prompt = "Frog astronaut"#input("New text prompt:")

        text_prompt = None
        try:
            text_prompt = input('\nIf you would like the robot to draw, type a description then press enter. Type nothing if you do not want to the robot to draw.\n:')
        except SyntaxError:
            # No input
            continue

        text_prompt = 'line drawing of ' + text_prompt + ', else empty, black and white'

        curr_canvas = painter.camera.get_canvas()
        print(curr_canvas.shape, curr_canvas.max())
        dark_inds = curr_canvas.mean(axis=2) < 0.75*255
        curr_canvas[dark_inds] = 5
        curr_canvas_pil = Image.fromarray(curr_canvas.astype(np.uint8)).resize((512,512))
        curr_canvas_pil = curr_canvas_pil.convert("L").convert('RGB')
        # plt.imshow(curr_canvas_pil)
        # plt.show()

        with torch.no_grad():
            target_img = sd_interactive_pipeline(
                text_prompt, curr_canvas_pil, num_inference_steps=20, 
                # generator=generator,
                num_images_per_prompt=1,
                # controlnet_conditioning_scale=1.4,
            ).images[0]

        # plt.imshow(target_img)
        # plt.show()
        target_img = torch.from_numpy(np.array(target_img)).permute(2,0,1).float().to(device)[None] / 255.
        target_img = Resize((h,w))(target_img)
        # print('target_img', target_img.shape, target_img.max())
        opt.writer.add_image('images/{}_2_target_from_controlnet_{}'.format(i, text_prompt), format_img(target_img), 0)

        # Start Planning
        painting = plan_from_image(opt, target_img, current_canvas) 

        # Export the strokes
        f = open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), "w")
        f.write(painting.to_csv())
        f.close()

        opt.writer.add_image('images/{}_3_planned_drawing'.format(i), 
                             format_img(painting(h,w, use_alpha=False)), 0)

        # Paint
        with open(os.path.join(painter.opt.cache_dir, "next_brush_strokes.csv"), 'r') as fp:
            from paint_planner import parse_csv_line_continuous
            instructions = [parse_csv_line_continuous(line, painter, None) for line in fp.readlines()] 
            n_instr = len(instructions)
            for instruction in tqdm(instructions[:], desc="Painting"):
                x, y, r, length, thickness, bend, alpha, color, color_ind, color_discrete = instruction
                # color = colors[color_ind].copy()

                # Convert the canvas proportion coordinates to meters from robot
                x, y = float(x) / painter.opt.CANVAS_WIDTH_PIX, 1 - (float(y) / painter.opt.CANVAS_HEIGHT_PIX)
                x, y = min(max(x,0.),1.), min(max(y,0.),1.) #safety
                x,y,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

                # Paint the brush stroke
                s = simple_parameterization_to_real(length, bend, thickness, alpha=alpha)
                # s.paint(painter, x, y, r * (2*3.14/360))
                s.angled_paint(painter, x, y, r * (2*3.14/360))

        painter.to_neutral()


        current_canvas = painter.camera.get_canvas()
        current_canvas = torch.from_numpy(current_canvas).permute(2,0,1).float().to(device)[None] / 255.
        current_canvas = Resize((h,w))(current_canvas)
        opt.writer.add_image('images/{}_4_canvas_after_drawing'.format(i), format_img(current_canvas), 0)
    
    painter.to_neutral()

    painter.robot.good_night_robot()


##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

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

from torch_painting_models_continuous import *
from style_loss import compute_style_loss
from sketch_loss.sketch_loss import compute_sketch_loss, compute_canny_loss

from clip_loss import clip_conv_loss, clip_model, clip_text_loss, clip_model_16, clip_fc_loss
import clip
import kornia as K

from paint_utils import to_video
from paint_utils3 import *

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

def parse_objective(objective_type, objective_data, p, weight=1.0):
    ''' p is the rendered painting '''
    global local_it
    if objective_type == 'text':
        return clip_text_loss(p, objective_data, opt.num_augs)[0] * weight
    elif objective_type == 'style':
        return compute_style_loss(p, objective_data) * weight
    elif objective_type == 'clip_conv_loss':
        return clip_conv_loss(p, objective_data) * weight
    elif objective_type == 'l2':
        return ((p - objective_data)**2).mean() * weight
    elif objective_type == 'clip_fc_loss':
        return clip_fc_loss(p, objective_data, opt.num_augs)[0] * weight
    elif objective_type == 'sketch':
        local_it += 1
        return compute_sketch_loss(objective_data, p, writer=writer, it=local_it) * weight
        # return compute_sketch_loss(objective_data, p, comparator=clip_conv_loss) * weight
        def clip_sketch_comparison(sketch, p):
            return clip_conv_loss(K.color.grayscale_to_rgb(sketch), K.color.grayscale_to_rgb(p))
        return compute_sketch_loss(objective_data, p,
            comparator=clip_sketch_comparison, writer=writer, it=local_it) * weight
    elif objective_type == 'canny':
        local_it += 1
        return compute_canny_loss(objective_data, p, writer=writer, it=local_it) * weight
        # def clip_sketch_comparison(sketch, p):
        #     return clip_conv_loss(K.color.grayscale_to_rgb(sketch), K.color.grayscale_to_rgb(p))
        # return compute_canny_loss(objective_data, p,
        #     comparator=clip_sketch_comparison, writer=writer, it=local_it) * weight


    else:
        print('don\'t know what objective')
        1/0

def plan(opt):
    global colors
    painting = random_init_painting(current_canvas, opt.num_strokes)

    # Do initilization objective(s)
    painting.to(device)

    if opt.init_objective:
        optim = torch.optim.RMSprop(painting.parameters(), lr=opt.init_lr) # Coarse optimization
        for j in tqdm(range(opt.init_optim_iter), desc="Initializing"):
            optim.zero_grad()
            p = painting(h, w, use_alpha=False)
            loss = 0
            for k in range(len(opt.init_objective)):
                loss += parse_objective(opt.init_objective[k], 
                    init_objective_data[k], p[:,:3], weight=opt.init_objective_weight[k])

            loss.backward()
            optim.step()
            painting.validate()
            optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * 0.95
            # painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            painting = randomize_brush_stroke_order(painting)
            log_progress(painting, title='init_optimization', log_freq=opt.log_frequency)#, force_log=True)
            

    # Intermediate optimization. Do it a few times, and pick the best
    init_painting = copy.deepcopy(painting.cpu())
    best_init_painting, best_init_loss = init_painting, 999999999
    painting.to(device)
    for attempt in range(opt.n_inits):
        painting = copy.deepcopy(init_painting).to(device)
        # optim = torch.optim.RMSprop(painting.parameters(), lr=5e-3) # Coarse optimization
        optims = painting.get_optimizers(multiplier=opt.lr_multiplier)
        for j in tqdm(range(opt.intermediate_optim_iter), desc="Intermediate Optimization"):
            #optim.zero_grad()
            for o in optims: o.zero_grad()
            p = painting(h, w, use_alpha=False)
            loss = 0
            for k in range(len(opt.objective)):
                loss += parse_objective(opt.objective[k], 
                    objective_data[k], p[:,:3], weight=opt.objective_weight[k])
            loss.backward()
            # optim.step()
            for o in optims: o.step()
            painting.validate()
            for o in optims: o.param_groups[0]['lr'] = o.param_groups[0]['lr'] * 0.95
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            # make sure hidden strokes get some attention
            # painting = randomize_brush_stroke_order(painting)
            log_progress(painting, title='int_optimization_{}'.format(attempt), log_freq=opt.log_frequency)#, force_log=True)
            
        if loss.item() < best_init_loss:
            best_init_loss = loss.item()
            best_init_painting = copy.deepcopy(painting.cpu())
            painting.to(device)
            print('best_painting {}'.format(attempt))
    painting = best_init_painting.to(device)

    # Create the plan
    position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt \
                = painting.get_optimizers(multiplier=opt.lr_multiplier)
    optims = (position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt)
    for i in tqdm(range(opt.optim_iter), desc='Optimizing {} Strokes'.format(str(len(painting.brush_strokes)))):
        for o in optims: o.zero_grad()

        p = painting(h, w, use_alpha=False)
        
        loss = 0
        for k in range(len(opt.objective)):
            loss += parse_objective(opt.objective[k], 
                objective_data[k], p[:,:3], weight=opt.objective_weight[k])
        loss.backward()

        position_opt.step()
        rotation_opt.step()
        bend_opt.step()
        length_opt.step()
        thickness_opt.step()
        if i < .8*opt.optim_iter: color_opt.step()

        # position_opt.param_groups[0]['lr'] = position_opt.param_groups[0]['lr'] * 0.99
        # rotation_opt.param_groups[0]['lr'] = rotation_opt.param_groups[0]['lr'] * 0.99
        # color_opt.param_groups[0]['lr'] = color_opt.param_groups[0]['lr'] * 0.99
        # bend_opt.param_groups[0]['lr'] = bend_opt.param_groups[0]['lr'] * 0.99
        # length_opt.param_groups[0]['lr'] = length_opt.param_groups[0]['lr'] * 0.99
        # thickness_opt.param_groups[0]['lr'] = thickness_opt.param_groups[0]['lr'] * 0.99

        painting.validate()

        painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
        
        if i < 0.3*opt.optim_iter and i %3 == 0:
            # make sure hidden strokes get some attention
            painting = randomize_brush_stroke_order(painting)

        if (i % 10 == 0 and i > (0.5*opt.optim_iter)) or i > 0.9*opt.optim_iter:
            if opt.use_colors_from is None:
                # Cluster the colors from the existing painting
                colors = painting.cluster_colors(opt.n_colors)

            discretize_colors(painting, colors)
        log_progress(painting, log_freq=opt.log_frequency)#, force_log=True)

    if opt.use_colors_from is None:
        colors = painting.cluster_colors(opt.n_colors)
    with open(os.path.join(opt.cache_dir, 'colors_updated.npy'), 'wb') as f:
        np.save(f, (colors.detach().cpu().numpy()*255).astype(np.uint8))


    discretize_colors(painting, colors)
    painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
    log_progress(painting, force_log=True, log_freq=opt.log_frequency)
    # save_painting_strokes(painting, opt)
    return painting

def adapt(opt):
    # Load available colors
    with open(os.path.join(opt.cache_dir, 'colors_updated.npy'), 'rb') as f:
        colors = (torch.from_numpy(np.array(np.load(f))) / 255.).to(device)

    # Load past plan
    with open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), 'r') as fp:
        instructions = [parse_csv_line_continuous(line) for line in fp.readlines()] 
    brush_strokes = []
    for instruction in instructions[int(opt.remove_prop*opt.strokes_before_adapting):]:
        x, y, r, length, thickness, bend, color = instruction
        brush_strokes.append(BrushStroke( 
            a=r, xt=x*2-1, yt=y*2-1, color=torch.from_numpy(color).float(),
            stroke_length=torch.ones(1)*length,
            stroke_z=torch.ones(1)*thickness,
            stroke_bend=torch.ones(1)*bend))

    painting = Painting(0, background_img=current_canvas, brush_strokes=brush_strokes).to(device)
    
    with torch.no_grad():
        p = painting(h,w, use_alpha=False)
        opt.writer.add_image('images/plan_update{}'.format(opt.global_it), format_img(p), 0)

    painting.validate()
    
    # Optimize all brush strokes
    position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt \
            = painting.get_optimizers(multiplier=opt.lr_multiplier*.25)
    # print(opt.objective)
    for j in tqdm(range(opt.adapt_optim_iter), desc='Optimizing {} Strokes'.format(str(len(painting.brush_strokes)))):
        position_opt.zero_grad()
        rotation_opt.zero_grad()
        color_opt.zero_grad()
        bend_opt.zero_grad()
        length_opt.zero_grad()
        thickness_opt.zero_grad()

        p = painting(h,w, use_alpha=False)
        loss = 0
        for k in range(len(opt.objective)):
            loss += parse_objective(opt.objective[k], 
                objective_data[k], p[:,:3], weight=opt.objective_weight[k])
        loss.backward()

        position_opt.step()
        rotation_opt.step()
        if j < .8*opt.adapt_optim_iter: color_opt.step()
        bend_opt.step()
        length_opt.step()
        thickness_opt.step()

        position_opt.param_groups[0]['lr'] = position_opt.param_groups[0]['lr'] * 0.99
        rotation_opt.param_groups[0]['lr'] = rotation_opt.param_groups[0]['lr'] * 0.99
        color_opt.param_groups[0]['lr'] = color_opt.param_groups[0]['lr'] * 0.99
        bend_opt.param_groups[0]['lr'] = bend_opt.param_groups[0]['lr'] * 0.99
        length_opt.param_groups[0]['lr'] = length_opt.param_groups[0]['lr'] * 0.99
        thickness_opt.param_groups[0]['lr'] = thickness_opt.param_groups[0]['lr'] * 0.99

        painting.validate()

        if j%2 == 0 and j > 0.5*opt.adapt_optim_iter:
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            discretize_colors(painting, colors)
        # log_progress(painting)#, force_log=True)

        if j%5 == 0 or j == opt.adapt_optim_iter-1:
            opt.writer.add_image('images/plan_update{}'.format(opt.global_it), format_img(p), j+1)

    painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
    discretize_colors(painting, colors)
    painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)

    with torch.no_grad():
        p = painting(h,w, use_alpha=False)
        opt.writer.add_image('images/plan_update{}'.format(opt.global_it), format_img(p), opt.adapt_optim_iter+1)

    return painting 

def load_objectives_data(opt):
    # Load Initial objective data
    global init_objective_data
    init_objective_data = [] 
    for i in range(len(opt.init_objective) if opt.init_objective else 0):
        if opt.init_objective[i] == 'text':
            with torch.no_grad():
                text_features = clip_model.encode_text(clip.tokenize(opt.init_objective_data[i]).to(device))
                init_objective_data.append(text_features)
        else:
            # Must be an image
            img = load_img(opt.init_objective_data[i],h=h, w=w).to(device)/255.
            init_objective_data.append(img)
            opt.writer.add_image('target/init_input{}'.format(i), format_img(img), 0)

    # Load Objective data
    global objective_data
    objective_data = [] 
    for i in range(len(opt.objective)):
        if opt.objective[i] == 'text':
            with torch.no_grad():
                text_features = clip_model.encode_text(clip.tokenize(opt.objective_data[i]).to(device))
                objective_data.append(text_features)
        else:
            # Must be an image
            img = load_img(opt.objective_data[i],h=h, w=w).to(device)/255.
            objective_data.append(img)
            opt.writer.add_image('target/input{}'.format(i), format_img(img), 0)

if __name__ == '__main__':
    global opt
    opt = Options()
    opt.gather_options()

    # painting = calibrate(opt)

    opt.writer = create_tensorboard()

    global h, w, colors, current_canvas, text_features, style_img, sketch
    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    w = int((opt.max_height/h)*w)
    h = int(opt.max_height)
    # print('hw', h, w)

    colors = None
    if opt.use_colors_from is not None:
        colors = get_colors(cv2.resize(cv2.imread(opt.use_colors_from)[:,:,::-1], (256, 256)), 
                n_colors=opt.n_colors)
        opt.writer.add_image('paint_colors/using_colors_from_input', save_colors(colors), 0)

    # Get the background of painting to be the current canvas
    if os.path.exists(os.path.join(opt.cache_dir, 'current_canvas.jpg')):
        current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg'), h=h, w=w).to(device)/255.
    else:
        current_canvas = torch.ones(1,3,h,w).to(device)
    load_objectives_data(opt)

    # Start Planning
    if opt.generate_whole_plan or not opt.adaptive:
        painting = plan(opt)
    else:
        painting = adapt(opt)

    to_video(plans, fn=os.path.join(opt.plan_gif_dir,'sim_canvases{}.mp4'.format(str(time.time()))))

    # Export the strokes
    f = open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), "w")
    f.write(painting.to_csv())
    f.close()
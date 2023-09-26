
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

# from torch_painting_models_continuous_concerted import *
from painting import *
from losses.style_loss import compute_style_loss
from losses.sketch_loss.sketch_loss import compute_sketch_loss, compute_canny_loss
from losses.audio_loss.audio_loss import compute_audio_loss, load_audio_file
from losses.emotion_loss.emotion_loss import emotion_loss
from losses.face.face_loss import face_loss, parse_face_data
from losses.stable_diffusion.stable_diffusion_loss2 import stable_diffusion_loss, encode_text_stable_diffusion
from losses.speech2emotion.speech2emotion import speech2emotion, speech2text

from losses.clip_loss import clip_conv_loss, clip_model, clip_text_loss, clip_model_16, clip_fc_loss
import clip
from clip_attn.clip_attn import get_attention
import kornia as K

from paint_utils import to_video
from paint_utils3 import *
from torchvision.utils import save_image


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
    elif objective_type == 'emotion':
        return emotion_loss(p, objective_data, opt.num_augs)[0] * weight
    elif objective_type == 'face':
        return face_loss(p, objective_data, opt.num_augs) * weight
    elif objective_type == 'stable_diffusion':
        # return sd_loss.stable_diffusion_loss(p, objective_data) * weight
        return stable_diffusion_loss(p, objective_data) * weight
    elif objective_type == 'speech':
        return emotion_loss(p, objective_data[0], opt.num_augs)[0] * weight \
            + clip_text_loss(p, objective_data[1], opt.num_augs)[0] * weight
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
    elif objective_type == 'audio':
        return compute_audio_loss(opt, objective_data, p)[0] * weight


    else:
        print('don\'t know what objective')
        1/0


def plan_from_image(opt):
    global colors
    painting = random_init_painting(current_canvas, opt.num_strokes, ink=opt.ink)

    target_img = objective_data[0]
    painting = initialize_painting(opt.num_strokes, target_img, current_canvas, opt.ink)
    
    attn = get_attention(target_img)
    opt.writer.add_image('target/attention', format_img(torch.from_numpy(attn)[None,None,:,:]), 0)
    if opt.caption:
        sd_encoded_text = encode_text_stable_diffusion(opt.caption)

    stroke_batch_size = 100#64
    iters_per_batch =  300

    painting = initialize_painting(0, target_img, current_canvas, opt.ink)
    painting.to(device)

    c = 0
    total_its = (opt.num_strokes/stroke_batch_size)*iters_per_batch
    for i in (range(0, opt.num_strokes, stroke_batch_size)):#, desc="Initializing"):
        painting = add_strokes_to_painting(painting, painting(h,w)[:,:3], stroke_batch_size, 
                                           target_img, current_canvas, opt.ink)
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
            loss += parse_objective('l2', target_img, p[:,:3], weight=1-t)
            loss += parse_objective('clip_conv_loss', target_img, p[:,:3], weight=t)
            # if i > opt.num_strokes/2 and opt.caption is not None:
            # loss += parse_objective('stable_diffusion', sd_encoded_text, p[:,:3], weight=t*2)

            # loss += (torch.abs(2-alphas)).mean() * 0.5#opt.fill_weight

            loss.backward()

            for o in optims: o.step() if o is not None else None
            painting.validate()
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
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


    if opt.use_colors_from is None:
        colors = painting.cluster_colors(opt.n_colors)
    with open(os.path.join(opt.cache_dir, 'colors_updated.npy'), 'wb') as f:
        np.save(f, (colors.detach().cpu().numpy()*255).astype(np.uint8))


    discretize_colors(painting, colors)
    painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
    log_progress(painting, force_log=True, log_freq=opt.log_frequency)

    return painting

def plan(opt):
    global colors
    painting = random_init_painting(current_canvas, opt.num_strokes, ink=opt.ink)
    # painting = Painting(opt.num_strokes, background_img=current_canvas)

    # Do initilization objective(s)
    painting.to(device)

    if opt.init_objective:
        optim = torch.optim.RMSprop(painting.parameters(), lr=opt.init_lr) # Coarse optimization
        for j in tqdm(range(opt.init_optim_iter), desc="Initializing"):
            optim.zero_grad()
            p, alphas = painting(h, w, use_alpha=False, return_alphas=True)
            loss = 0
            for k in range(len(opt.init_objective)):
                loss += parse_objective(opt.init_objective[k], 
                    init_objective_data[k], p[:,:3], weight=opt.init_objective_weight[k])

            loss += (1-alphas).mean() * opt.fill_weight

            loss.backward()
            optim.step()
            painting.validate()
            optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * 0.95
            # painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            if not opt.ink: 
                painting = randomize_brush_stroke_order(painting)
            log_progress(painting, title='init_optimization', log_freq=opt.log_frequency)#, force_log=True)
            

    # Intermediate optimization. Do it a few times, and pick the best
    init_painting = copy.deepcopy(painting.cpu())
    best_init_painting, best_init_loss = init_painting, 999999999
    painting.to(device)
    for attempt in range(opt.n_inits):
        painting = copy.deepcopy(init_painting).to(device)
        # optim = torch.optim.RMSprop(painting.parameters(), lr=5e-3) # Coarse optimization
        optims = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)
        for j in tqdm(range(opt.intermediate_optim_iter), desc="Intermediate Optimization"):
            #optim.zero_grad()
            for o in optims: o.zero_grad() if o is not None else None
            p, alphas = painting(h, w, use_alpha=True, return_alphas=True)
            loss = 0
            for k in range(len(opt.objective)):
                loss += parse_objective(opt.objective[k], 
                    objective_data[k], p[:,:3], weight=opt.objective_weight[k])

            loss += (1-alphas).mean() * opt.fill_weight

            loss.backward()
            # optim.step()
            for o in optims: o.step() if o is not None else None
            painting.validate()
            for o in optims: o.param_groups[0]['lr'] = o.param_groups[0]['lr'] * 0.95 if o is not None else None
            if not opt.ink:
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
                = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)
    optims = (position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt)
    for i in tqdm(range(opt.optim_iter), desc='Optimizing {} Strokes'.format(str(len(painting.brush_strokes)))):
        for o in optims: o.zero_grad() if o is not None else None

        p, alphas = painting(h, w, use_alpha=False, return_alphas=True)
        
        loss = 0
        for k in range(len(opt.objective)):
            loss += parse_objective(opt.objective[k], 
                objective_data[k], p[:,:3], weight=opt.objective_weight[k])
        #loss += (1-alphas).mean() * opt.fill_weight
        loss += torch.abs(1-alphas).mean() * opt.fill_weight
        loss.backward()

        position_opt.step()
        rotation_opt.step()
        bend_opt.step()
        length_opt.step()
        thickness_opt.step()
        if i < .8*opt.optim_iter: color_opt.step() if color_opt is not None else None

        # position_opt.param_groups[0]['lr'] = position_opt.param_groups[0]['lr'] * 0.99
        # rotation_opt.param_groups[0]['lr'] = rotation_opt.param_groups[0]['lr'] * 0.99
        # color_opt.param_groups[0]['lr'] = color_opt.param_groups[0]['lr'] * 0.99
        # bend_opt.param_groups[0]['lr'] = bend_opt.param_groups[0]['lr'] * 0.99
        # length_opt.param_groups[0]['lr'] = length_opt.param_groups[0]['lr'] * 0.99
        # thickness_opt.param_groups[0]['lr'] = thickness_opt.param_groups[0]['lr'] * 0.99

        painting.validate()

        if not opt.ink:
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
        
        if i < 0.3*opt.optim_iter and i %3 == 0:
            # make sure hidden strokes get some attention
            if not opt.ink:
                painting = randomize_brush_stroke_order(painting)

        if (i % 10 == 0 and i > (0.5*opt.optim_iter)) or i > 0.9*opt.optim_iter:
            if opt.use_colors_from is None:
                # Cluster the colors from the existing painting
                if not opt.ink:
                    colors = painting.cluster_colors(opt.n_colors)

            if not opt.ink:
                discretize_colors(painting, colors)
        log_progress(painting, log_freq=opt.log_frequency)#, force_log=True)

        # Save paintings every 150 steps 
        if i % 150 == 0:
            if not os.path.exists(opt.output_dir):
                os.makedirs(opt.output_dir)
            save_image(p, os.path.join(opt.output_dir, 'painting_{}.png'.format(i)))



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

def parse_emotion_data(s):
    weights_str = s.split(',')
    if len(weights_str) != 9:
        print('you must specify weights for the 9 emotions. You did:', len(weights_str))
        1/0

    weights = [float(i) for i in weights_str]
    weights = torch.tensor(weights).float().to(device)
    return weights.unsqueeze(0)

def load_objectives_data(opt):
    # Load Initial objective data
    global init_objective_data
    init_objective_data = [] 
    for i in range(len(opt.init_objective) if opt.init_objective else 0):
        if opt.init_objective[i] == 'text':
            with torch.no_grad():
                text_features = clip_model.encode_text(clip.tokenize(opt.init_objective_data[i]).to(device))
                init_objective_data.append(text_features)
        elif opt.init_objective[i] == 'audio':
            with torch.no_grad():
                init_objective_data.append(load_audio_file(opt.init_objective_data[i]))
        elif opt.init_objective[i] == 'emotion':
            with torch.no_grad():
                init_objective_data.append(parse_emotion_data(opt.init_objective_data[i]))
        elif opt.init_objective[i] == 'face':
            with torch.no_grad():
                img = load_img(opt.init_objective_data[i],h=h, w=w).to(device)/255.
                init_objective_data.append(parse_face_data(img))
        elif opt.init_objective[i] == 'stable_diffusion':
            with torch.no_grad():
                init_objective_data.append(encode_text_stable_diffusion(opt.init_objective_data[i]))
        elif opt.init_objective[i] == 'speech':
            with torch.no_grad():
                emotion = torch.tensor(speech2emotion(opt.init_objective_data[i])).float().to(device)
                speech_text = speech2text(opt.init_objective_data[i])
                text_features = clip_model.encode_text(clip.tokenize(speech_text).to(device))
                init_objective_data.append([emotion, text_features])
        else:
            # Must be an image
            img = load_img(opt.init_objective_data[i],h=h, w=w).to(device)/255.
            init_objective_data.append(img)
            opt.writer.add_image('target/init_input{}'.format(i), format_img(img), 0)

    # Load Objective data
    global objective_data
    objective_data = [] 
    if not opt.objective:
        print('\n\nNo objectives. Are you sure?\n\n')
    for i in range(len(opt.objective) if opt.objective else 0):
        if opt.objective[i] == 'text':
            with torch.no_grad():
                text_features = clip_model.encode_text(clip.tokenize(opt.objective_data[i]).to(device))
                objective_data.append(text_features)
        elif opt.objective[i] == 'audio':
            with torch.no_grad():
                objective_data.append(load_audio_file(opt.objective_data[i]))
        elif opt.objective[i] == 'emotion':
            with torch.no_grad():
                objective_data.append(parse_emotion_data(opt.objective_data[i]))
        elif opt.objective[i] == 'face':
            with torch.no_grad():
                img = load_img(opt.objective_data[i],h=h, w=w).to(device)/255.
                objective_data.append(parse_face_data(img))
        elif opt.objective[i] == 'stable_diffusion':
            with torch.no_grad():
                objective_data.append(encode_text_stable_diffusion(opt.objective_data[i]))
        elif opt.objective[i] == 'speech':
            with torch.no_grad():
                emotion = torch.tensor(speech2emotion(opt.objective_data[i])).float().to(device)
                speech_text = speech2text(opt.objective_data[i])
                text_features = clip_model.encode_text(clip.tokenize(speech_text).to(device))
                objective_data.append([emotion, text_features])
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

    opt.writer = create_tensorboard(log_dir=opt.tensorboard_dir)

    global h, w, colors, current_canvas, text_features, style_img, sketch
    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    w = int((opt.max_height/h)*w)
    h = int(opt.max_height)

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
        painting = plan_from_image(opt) if opt.paint_from_image else plan(opt)
    else:
        painting = adapt(opt)

    to_video(plans, fn=os.path.join(opt.plan_gif_dir,'sim_canvases{}.mp4'.format(str(time.time()))))
    with torch.no_grad():
        save_image(painting(h*4,w*4, use_alpha=False), os.path.join(opt.plan_gif_dir, 'init_painting_plan{}.png'.format(str(time.time()))))

    # Export the strokes
    f = open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), "w")
    f.write(painting.to_csv())
    f.close()
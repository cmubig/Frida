
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################


import copy
import numpy as np
import torch
from tqdm import tqdm
import os


from losses.style_loss import compute_style_loss
from losses.audio_loss.audio_loss import compute_audio_loss, load_audio_file
from losses.emotion_loss.emotion_loss import emotion_loss
from losses.face.face_loss import face_loss, parse_face_data
from losses.stable_diffusion.stable_diffusion_loss2 import stable_diffusion_loss, encode_text_stable_diffusion
from losses.speech2emotion.speech2emotion import speech2emotion, speech2text

from losses.dino_loss import dino_loss
from losses.clip_loss import clip_conv_loss, clip_model, clip_text_loss, clip_fc_loss
import clip

from paint_utils3 import discretize_colors, format_img, load_img, randomize_brush_stroke_order, sort_brush_strokes_by_color, sort_brush_strokes_by_location

# from paint_utils3 import *

def flip_img(img):
    return torch.flip(img, dims=(2,3))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

# Utilities

writer = None
local_it = 0 
plans = []

def log_progress(painting, opt, log_freq=5, force_log=False, title='plan'):
    global local_it, plans
    local_it +=1
    if ((local_it-1) % log_freq==0) or force_log:
        with torch.no_grad():
            #np_painting = painting(h,w, use_alpha=False).detach().cpu().numpy()[0].transpose(1,2,0)
            #opt.writer.add_image('images/{}'.format(title), np.clip(np_painting, a_min=0, a_max=1), local_it)
            p = painting(opt.h_render,opt.w_render, use_alpha=False)
            p = flip_img(p)
            p = format_img(p)
            opt.writer.add_image('images/{}'.format(title), p, local_it)
            
            plans.append((p*255.).astype(np.uint8))

def parse_objective(objective_type, objective_data, p, weight=1.0, num_augs=30):
    ''' p is the rendered painting '''
    global local_it
    if objective_type == 'text':
        return clip_text_loss(p, objective_data, num_augs)[0] * weight
    elif objective_type == 'style':
        return compute_style_loss(p, objective_data) * weight
    elif objective_type == 'clip_conv_loss':
        return clip_conv_loss(p, objective_data) * weight
    elif objective_type == 'dino':
        return dino_loss(p, objective_data) * weight
    elif objective_type == 'l2':
        return ((p - objective_data)**2).mean() * weight
    elif objective_type == 'clip_fc_loss':
        return clip_fc_loss(p, objective_data, num_augs)[0] * weight
    elif objective_type == 'emotion':
        return emotion_loss(p, objective_data, num_augs)[0] * weight
    elif objective_type == 'face':
        return face_loss(p, objective_data, num_augs) * weight
    elif objective_type == 'stable_diffusion':
        # return sd_loss.stable_diffusion_loss(p, objective_data) * weight
        return stable_diffusion_loss(p, objective_data) * weight
    elif objective_type == 'speech':
        return emotion_loss(p, objective_data[0], num_augs)[0] * weight \
            + clip_text_loss(p, objective_data[1], num_augs)[0] * weight
    elif objective_type == 'audio':
        return compute_audio_loss(objective_data, p)[0] * weight
    else:
        print('don\'t know what objective')
        1/0


def parse_emotion_data(s):
    weights_str = s.split(',')
    if len(weights_str) != 9:
        print('you must specify weights for the 9 emotions. You did:', len(weights_str))
        1/0

    weights = [float(i) for i in weights_str]
    weights = torch.tensor(weights).float().to(device)
    return weights.unsqueeze(0)

def load_objectives_data(opt):
    # Load Objective data
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
                img = load_img(opt.objective_data[i],h=opt.h_render, w=opt.w_render).to(device)/255.
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
            img = load_img(opt.objective_data[i],h=opt.h_render, w=opt.w_render).to(device)/255.
            objective_data.append(img)
            opt.writer.add_image('target/input{}'.format(i), format_img(img), 0)
    opt.objective_data_loaded = objective_data

def optimize_painting(opt, painting, optim_iter, color_palette=None,
                      change_color=True, shuffle_strokes=True, log_title='plan'):
    """
    kwargs:
        color_palette: if None, then it creates a new one
    """
    # torch.autograd.set_detect_anomaly(True)
    use_input_palette =  color_palette is not None
    if len(painting) == 0: return painting, color_palette

    position_opt, rotation_opt, color_opt, path_opt \
                = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)
    if not change_color:
        color_opt.param_groups[0]['lr'] = 0.0
    optims = (position_opt, rotation_opt, color_opt, path_opt)
    # optims = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)
    # Learning rate scheduling. Start low, middle high, end low
    og_lrs = [o.param_groups[0]['lr'] if o is not None else None for o in optims]

    for it in tqdm(range(optim_iter), desc='Optimizing {} Strokes'.format(str(len(painting.brush_strokes)))):
        for o in optims: o.zero_grad() if o is not None else None
        # print(painting.brush_strokes[0].path[0,:5])

        lr_factor = (1 - 2*np.abs(it/optim_iter - 0.5)) + 0.005
        # if it < 0.25*optim_iter:
        #     lr_factor += 0.3
        # lr_factor = (1 - np.abs(it/optim_iter)) + 0.001 # 1.001 -> 0.001
        for i_o in range(len(optims)):
            if optims[i_o] is not None:
                optims[i_o].param_groups[0]['lr'] = og_lrs[i_o]*lr_factor
                # if i_o == 3:
                #     print(optims[i_o].param_groups[0]['lr'])

        p, alphas = painting(opt.h_render, opt.w_render, use_alpha=False, return_alphas=True)
        
        loss = 0
        for k in range(len(opt.objective)):
            loss += parse_objective(opt.objective[k], 
                opt.objective_data_loaded[k], p[:,:3], 
                weight=opt.objective_weight[k],
                num_augs=opt.num_augs)
        #loss += (1-alphas).mean() * opt.fill_weight
        if opt.fill_weight > 0:
            loss += torch.abs(1-alphas).mean() * opt.fill_weight

        # if True:#it < 0.2*optim_iter:
        #     # loss = 0
        #     for bs in painting.brush_strokes:
        #         path = bs.get_path()
        #         #loss += path[:,2].sum()*100
        #         d = torch.abs(path[1:,2] - path[:-1,2]).sum()
        #         loss += d*0.1
            
        # stroke_latent_reg_loss_weight = 1.0
        # if stroke_latent_reg_loss_weight > 0:
        #     latent_reg_loss = 0
        #     for bs in painting.brush_strokes:
        #         latent = bs.latent
        #         latent_std = latent.std()
        #         latent_mean = latent.mean()
        #         if latent_std > 1.2:
        #             latent_reg_loss += latent_std
        #         if latent_mean > 0.25:
        #             latent_reg_loss += latent_mean 
        #         elif latent_mean < -0.25:
        #             latent_reg_loss += -1.0 * latent_mean 
        #     latent_reg_loss /= len(painting.brush_strokes)
        #     latent_reg_loss *= stroke_latent_reg_loss_weight
        #     if latent_reg_loss != 0:
        #         loss += latent_reg_loss
        
        path_loss_weight = 50.0
        if path_loss_weight > 0:
            path_loss = 0
            for bs in painting.brush_strokes:
                path = bs.path
                point_wise_dist = (path[:,1:,0]-path[:,:-1,0])**2 + (path[:,1:,1]-path[:,:-1,1])**2
                path_loss += point_wise_dist.sum()
            path_loss = path_loss / len(painting.brush_strokes)
            # print(path_loss)
            loss += path_loss * path_loss_weight
                
        
        loss.backward()

        for o in optims: o.step() if o is not None else None

        painting.validate(brush_width=(1-(it/optim_iter))*0.2+0.45)

        if it < 0.4*optim_iter:
            with torch.no_grad():
                for bs in painting.brush_strokes:
                    bs.color_transform *= 0

        # if it == optim_iter -1:#it%10==0 and it > 0.7*optim_iter:
        #     # Remove strokes that don't help the loss
        #     with torch.no_grad():
        #         full_painting = copy.deepcopy(painting)
        #         for i in range(len(full_painting)-1, 0, -1):
        #             partial_painting = copy.deepcopy(full_painting)
        #             partial_painting.remove(i)
        #             full_loss = 0
        #             p, alphas = full_painting(opt.h_render, opt.w_render, use_alpha=False, return_alphas=True)
        #             for k in range(len(opt.objective)):
        #                 full_loss += parse_objective(opt.objective[k], 
        #                     opt.objective_data_loaded[k], p[:,:3], 
        #                     weight=opt.objective_weight[k],
        #                     num_augs=opt.num_augs)
        #             partial_loss = 0
        #             p, alphas = partial_painting(opt.h_render, opt.w_render, use_alpha=False, return_alphas=True)
        #             for k in range(len(opt.objective)):
        #                 partial_loss += parse_objective(opt.objective[k], 
        #                     opt.objective_data_loaded[k], p[:,:3], 
        #                     weight=opt.objective_weight[k],
        #                     num_augs=1000)
        #             if partial_loss < full_loss:
        #                 full_painting = copy.deepcopy(partial_painting)
        #         print('strokes removed', len(full_painting) - len(painting))
        #         painting = copy.deepcopy(full_painting)
        #         position_opt, rotation_opt, color_opt, path_opt \
        #                     = painting.get_optimizers(multiplier=opt.lr_multiplier, ink=opt.ink)
        #         if not change_color:
        #             color_opt.param_groups[0]['lr'] = 0.0
        #         optims = (position_opt, rotation_opt, color_opt, path_opt)

        if not opt.ink and shuffle_strokes:
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
        elif opt.ink and shuffle_strokes:
            painting = sort_brush_strokes_by_location(painting)
        
        if it < 0.3*optim_iter and it %3 == 0 and not opt.ink and shuffle_strokes:
            # make sure hidden strokes get some attention
            painting = randomize_brush_stroke_order(painting)

        if (it % 10 == 0 and it > (0.5*optim_iter)) or it > 0.9*optim_iter:
            if opt.use_colors_from is None:
                # Cluster the colors from the existing painting
                if not opt.ink and not use_input_palette:
                    color_palette = painting.cluster_colors(opt.n_colors)

            if not opt.ink:
                discretize_colors(painting, color_palette)
        log_progress(painting, opt, log_freq=opt.log_frequency, title=log_title)#, force_log=True)

    painting.validate(brush_width=0.45)
    if not use_input_palette and not opt.ink:
        color_palette = painting.cluster_colors(opt.n_colors)

    if not opt.ink:
        with open(os.path.join(opt.cache_dir, 'colors_updated.npy'), 'wb') as f:
            np.save(f, (color_palette.detach().cpu().numpy()*255).astype(np.uint8))
        discretize_colors(painting, color_palette)
        if shuffle_strokes:
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
    if opt.ink and shuffle_strokes:
        painting = sort_brush_strokes_by_location(painting)    
    
    log_progress(painting, opt, force_log=True, log_freq=opt.log_frequency, title=log_title)

    latent_std = 0
    latent_mean = 0
    for bs in painting.brush_strokes:
        latent = bs.latent
        latent_std += latent.std()
        latent_mean += latent.mean()
    latent_std /= len(painting.brush_strokes)
    latent_mean /= len(painting.brush_strokes)
    print('latent', latent_mean.cpu().detach().numpy(), latent_std.cpu().detach().numpy())

    return painting, color_palette



# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import requests
from PIL import Image
import io
import cv2
from tqdm import tqdm
import os
import PIL.Image, PIL.ImageDraw
from io import BytesIO
import lpips
import copy
import colour
import random
import gzip

import sys
import subprocess

from diffvg_helper import *

from options import Options
from tensorboard import TensorBoard

from torch_painting_models import *
# from style_loss import compute_style_loss

# from clip_loss import clip_conv_loss, clip_model, clip_text_loss
# import clip
# import kornia as K

from plan_all_strokes import load_img, get_colors, show_img, load_brush_strokes, log_progress, log_painting#, loss_fcn
from model_brush_strokes3 import get_stroke_width_translation_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')
print(device, 'in plan_all_strokes')


def sort_brush_strokes(shapes, shape_groups):
    #t = [x for (y,x) in sorted(zip(shape_groups, shapes), key=lambda pair: pair[0].stroke_color.mean(), reverse=True)]
    #shape_groups, shapes = t[0], t[1]
    # print('asdf')
    # print(shape_groups, shapes)
    shape_groups, shapes = zip(*sorted(zip(shape_groups, shapes), key=lambda pair: pair[0].stroke_color.mean(), reverse=True))
    
    # print('afsfsf')
    # print(shape_groups, shapes)
    shape_groups = fix_shape_group_ids(shape_groups)
    # print('ffff')
    return shapes, shape_groups
def plan_all_strokes(opt, min_stroke_width=4, max_stroke_width=7, max_stroke_length=.05, num_passes=1, \
          num_paths=5, num_iter=50, debug=False):
    '''
    Perform StyleCLIPDraw using a given text prompt and target image
    '''
    h = sim_canvas_height
    w = sim_canvas_width

    max_stroke_length *= h
    min_stroke_width *= h 
    max_stroke_width *= h 
    print(min_stroke_width, max_stroke_width, 'min_stroke_width, max_stroke_width')
    # min_stroke_width, max_stroke_width = 3, 7

    # target = load_img(opt.target,
    #     h=h, w=w).to(device)/255.
    target = load_img(os.path.join(opt.cache_dir, 'target_discrete.jpg'),
        h=h, w=w).to(device)/255.

    current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg'), h=h, w=w).to(device)/255.
    background_img = current_canvas
    total_shapes, total_shape_groups = [], []
    for layer_i in range(num_passes):
        torch.cuda.empty_cache()
        # Initialize Random Curves
        # print('1')
        # shapes, shape_groups, points_vars, color_vars, width_vars = initialize_curves(num_paths, w, h)
        shapes, shape_groups, points_vars, color_vars, width_vars = initialize_curves_grid(w, h, target,  num_strokes_x=int(num_paths**.5), num_strokes_y=int(num_paths**.5))
        shape_groups = fix_shape_group_ids(shape_groups)
        # print('2')
        # Optimizers
        points_optim = torch.optim.Adam(points_vars, lr=.1)
        width_optim  = torch.optim.Adam(width_vars,  lr=.1)
        color_optim  = torch.optim.Adam(color_vars,  lr=0.01)

        l1loss = nn.L1Loss()
        # print('2.5')
        # Run the main optimization loop
        for t in tqdm(range(num_iter)):
            points_optim.zero_grad()
            color_optim.zero_grad()
            width_optim.zero_grad()
            # print(3)
            # t = torch.cuda.get_device_properties(0).total_memory
            # r = torch.cuda.memory_reserved(0)
            # a = torch.cuda.memory_allocated(0)
            # f = r-a  # free inside reserved
            # print(t, r, a, f)
            img = render_drawing(shapes, shape_groups, w, h, t, background_img=background_img[0].permute((1,2,0)))
            # img = render_drawing(shapes, shape_groups, w, h, t, background_img=None).to(device)
            # print(4)
            loss = 0
            loss += l1loss(img, target) 

            # loss += loss_fcn(img, target,  use_clip_loss=True, use_style_loss=False, use_l2_loss=False)

            loss.backward()

            points_optim.step()
            color_optim.step()
            width_optim.step()
            
            for path in shapes:
                path.stroke_width.data.clamp_(min_stroke_width, max_stroke_width)
                path.points.data[:,0].clamp_(0, w)
                path.points.data[:,1].clamp_(0, h)
                path.points.data.nan_to_num_()
                # TODO: limit length of brush stroke
                # This below is wrong, it really messes stuff up. might work now with the dist> condition
                xs, ys = path.points.data[0,0], path.points.data[0,1]
                for p_i in range(1, len(path.points)):
                    x, y = path.points.data[p_i,0], path.points.data[p_i,1]
                    dist = ((x-xs)**2 + (y-ys)**2)**(.5)
                    if dist > max_stroke_length:
                        xn = x - (max_stroke_length/dist)*(x - xs)
                        yn = y - (max_stroke_length/dist)*(y - ys)
                        path.points.data[p_i,0], path.points.data[p_i,1] = xn, yn

            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)
                opac = max(0.8,t/num_iter)
                group.stroke_color.data[-1].clamp_(opac,opac)#(1.0, 1.0)
            
            points_optim.zero_grad()
            color_optim.zero_grad()
            width_optim.zero_grad()

            if t > (.5*num_iter) and t % 10 == 0:
                shapes, shape_groups = sort_brush_strokes(shapes, shape_groups)
                points_optim = torch.optim.Adam(points_vars, lr=.1)
                width_optim  = torch.optim.Adam(width_vars,  lr=.1)
                color_optim  = torch.optim.Adam(color_vars,  lr=0.01)

            if t % 1 == 0 or t == (num_iter-1):
                np_painting = img.detach().cpu().numpy()[0].transpose(1,2,0)
                opt.writer.add_image('images/grid_add_layer{}'.format(layer_i), 
                    np.clip(np_painting, a_min=0, a_max=1), t)

        with torch.no_grad():
            # Ideally, you'd render again, but this breaks things. Silently. No way to tell what's wrong.
            background_img = img.detach().clone()

        total_shapes += shapes
        total_shape_groups += shape_groups
        # print('stored shapes')
    
    fix_shape_group_ids(total_shape_groups)
    img = render_drawing(total_shapes, total_shape_groups, w, h, t, background_img=current_canvas[0].permute((1,2,0)))
    show_img(img)
    return total_shapes, total_shape_groups


def to_object(shapes, shape_groups, trans_model):
    # [{xs:np.array[x0,x1,x2,x3], ys:np.array[y0,y1,y2,y3], z:float, color:np.array(3)}]
    o = []
    for i in range(len(shapes)):
        obj = {}
        p = shapes[i].points.detach().cpu().numpy()
        print(p[:,0], p[:,1])
        obj['xs'] = np.array(p[:,0]) / sim_canvas_width
        obj['ys'] = np.array(p[:,1]) / sim_canvas_height
        obj['z'] = trans_model.sim_to_real(shapes[i].stroke_width/224.).detach().cpu().numpy()
        obj['color'] = shape_groups[i].stroke_color.detach().cpu().numpy()[:3]
        o.append(obj)
    return o # remove trailing newline

if __name__ == '__main__':
    global opt, sim_canvas_width, sim_canvas_height
    opt = Options()
    opt.gather_options()

    sim_canvas_height, sim_canvas_width = 224, int((14/11)*224)

    b = './painting'
    all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
    tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
    if '_planner' not in tensorboard_dir:
        tensorboard_dir += '_planner'

    writer = TensorBoard(tensorboard_dir)
    opt.writer = writer

    #trans_model, min_sim_z, max_sim_z = get_stroke_width_translation_model()
    # min_sim_z, max_sim_z = .01, .02
    exit_code = subprocess.call(['python3.7', '/home/frida/ros_ws/src/intera_sdk/SawyerPainter/scripts/model_brush_strokes3.py']+sys.argv[1:])
    with open(os.path.join(opt.cache_dir, 'min_sim_z.pkl'), 'rb') as f:
        min_sim_z = pickle.load(f, encoding="latin1")
    with open(os.path.join(opt.cache_dir, 'max_sim_z.pkl'), 'rb') as f:
        max_sim_z = pickle.load(f, encoding="latin1")
    with open(os.path.join(opt.cache_dir, 'max_stroke_length.pkl'), 'rb') as f:
        max_stroke_length = pickle.load(f, encoding="latin1")
    trans_model = WidthTranslation().to(device)
    trans_model.load_state_dict(torch.load(os.path.join(opt.cache_dir, 'trans_model.pkl')))
    trans_model.eval()
    # print(trans_model.a, trans_model.b, trans_model.e)
    # print('ey')

    if opt.prompt is not None:
        shapes, shape_groups = plan_all_strokes_text(opt, min_sim_z, max_sim_z, max_stroke_length)
    else:
        shapes, shape_groups = plan_all_strokes(opt, min_sim_z, max_sim_z, max_stroke_length)

    
    # Export the strokes
    obj = to_object(shapes, shape_groups, trans_model)
    with open(os.path.join(opt.cache_dir, 'next_brush_strokes_diffvg.pkl'),'wb') as f:
        pickle.dump(obj, f, protocol=2)
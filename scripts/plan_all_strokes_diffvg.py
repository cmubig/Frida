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

import pydiffvg

from options import Options
from tensorboard import TensorBoard

from torch_painting_models import *
from style_loss import compute_style_loss


from clip_loss import clip_conv_loss, clip_model, clip_text_loss
import clip
import kornia as K

from plan_all_strokes import load_img, get_colors, show_img, load_brush_strokes, log_progress, log_painting

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def initialize_curves(num_paths, canvas_width, canvas_height):
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(1,1)#1, 1)#3) #################
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points = num_control_points, 
                points = points, stroke_width = torch.tensor(random.random()*5.0+1.0), is_closed = False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, 
            stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
        shape_groups.append(path_group)
    
    points_vars = []
    color_vars = []
    width_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    return shapes, shape_groups, points_vars, color_vars, width_vars

def render_drawing(shapes, shape_groups,\
                   canvas_width, canvas_height, n_iter, save=False, no_grad=False, background_img=None):
    if no_grad:
        with torch.no_grad():
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                canvas_width, canvas_height, shapes, shape_groups)
            render = pydiffvg.RenderFunction.apply
            # render.requires_grad=False
            img = render(canvas_width, canvas_height, 2, 2, n_iter, background_img, *scene_args)
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])        
            # if save:
            #     pydiffvg.imwrite(img.cpu(), '/content/res/iter_{}.png'.format(int(n_iter)), gamma=1.0)
            img = img[:, :, :3]
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
            return img
    else:
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, n_iter, background_img, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])        
        # if save:
        #     pydiffvg.imwrite(img.cpu(), '/content/res/iter_{}.png'.format(int(n_iter)), gamma=1.0)
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        return img

def fix_shape_group_ids(shape_groups):
    with torch.no_grad():
        j = 0
        for shape_group in shape_groups:
            shape_group.shape_ids = torch.tensor([j])
            j += 1
    return shape_groups

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
def plan_all_strokes(opt,  num_passes=2, h=224,\
          num_paths=300, num_iter=400, debug=False):
    '''
    Perform StyleCLIPDraw using a given text prompt and target image
    '''
    w = int((14/11)*224)

    # target = load_img(opt.target,
    #     h=h, w=w).to(device)/255.
    target = load_img(os.path.join(opt.cache_dir, 'target_discrete.jpg'),
        h=h, w=w).to(device)/255.

    current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg'), h=h, w=w).to(device)/255.
    background_img = current_canvas
    total_shapes, total_shape_groups = [], []
    for layer_i in range(num_passes):
        # Initialize Random Curves
        # print('1')
        shapes, shape_groups, points_vars, color_vars, width_vars = initialize_curves(num_paths, w, h)
        shape_groups = fix_shape_group_ids(shape_groups)
        # print('2')
        # Optimizers
        points_optim = torch.optim.Adam(points_vars, lr=.1)
        width_optim  = torch.optim.Adam(width_vars,  lr=.1)
        color_optim  = torch.optim.Adam(color_vars,  lr=0.01)

        loss_fcn = nn.L1Loss()

        # Run the main optimization loop
        for t in tqdm(range(num_iter)):
            points_optim.zero_grad()
            color_optim.zero_grad()
            width_optim.zero_grad()

            img = render_drawing(shapes, shape_groups, w, h, t, background_img=background_img[0].permute((1,2,0)))

            loss = loss_fcn(img, target) 

            loss.backward()

            points_optim.step()
            color_optim.step()
            width_optim.step()
            
            for path in shapes:
                path.stroke_width.data.clamp_(2.0, 5.0)
                path.points.data[:,0].clamp_(0, w)
                path.points.data[:,1].clamp_(0, h)
                # TODO: limit length of brush stroke
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)
                group.stroke_color.data[-1].clamp_(1.0, 1.0)
            
            points_optim.zero_grad()
            color_optim.zero_grad()
            width_optim.zero_grad()

            if t > (.5*num_iter) and t % 10 == 0:
                shapes, shape_groups = sort_brush_strokes(shapes, shape_groups)
                points_optim = torch.optim.Adam(points_vars, lr=.1)
                width_optim  = torch.optim.Adam(width_vars,  lr=.1)
                color_optim  = torch.optim.Adam(color_vars,  lr=0.01)

            if t % 20 == 0 or t == (num_iter-1):
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
    return img

def to_csv(shapes, shape_groups):
    # x0,x1,x2,x3, y0,y1,y2,y3, z, r,g,b

if __name__ == '__main__':
    global opt
    opt = Options()
    opt.gather_options()

    b = './painting'
    all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
    tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
    if '_planner' not in tensorboard_dir:
        tensorboard_dir += '_planner'

    writer = TensorBoard(tensorboard_dir)
    opt.writer = writer

    if opt.prompt is not None:
        painting = plan_all_strokes_text(opt)
    else:
        painting = plan_all_strokes(opt)

    # Export the strokes
    # f = open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), "w")
    # f.write(painting.to_csv())
    # f.close()
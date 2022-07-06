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
import datetime

from options import Options
from tensorboard import TensorBoard

# from torch_painting_models import *
# from style_loss import compute_style_loss
from strokes import all_strokes


# from clip_loss import clip_conv_loss, clip_model, clip_text_loss
# import clip
# import kornia as K

from plan_all_strokes import load_brush_strokes, show_img

import pydiffvg

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def render_drawing(shapes, shape_groups,\
                   canvas_width, canvas_height, n_iter, save=False):
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, n_iter, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])        
    

    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
    return img

def init_diffvg_brush_stroke(h, w):

    cp = torch.tensor([
        [.5*w, .5*h],
        [.5*w+.01*w, .5*h],
        [.5*w+.02*w, .5*h],
        [.5*w+.03*w, .5*h],
    ])
    shape = pydiffvg.Path(
            num_control_points=torch.tensor([2], dtype = torch.int32),
            points=cp,
            stroke_width=torch.tensor(2.),
            is_closed=False
    )
    shape_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=None, 
            stroke_color=torch.tensor([.1,.1,.1, .9]))
    return shape, shape_group


class RealStrokeParameters(nn.Module):
    def __init__(self, xs, ys, zs):
        super(RealStrokeParameters, self).__init__()

        self.xs = nn.Parameter(xs)
        self.ys = nn.Parameter(ys)
        self.zs = nn.Parameter(zs)
    def forward(self):
        return torch.cat([self.xs, self.ys, self.zs]).unsqueeze(0)

# class RealToSimTranslation(nn.Module):
#     def __init__(self, canvas_width_pix, canvas_height_pix, canvas_width_meters, canvas_height_meters):
#         super(RealToSimTranslation, self).__init__()

#         self.fc_stroke_width0 = nn.Linear(4, 5)
#         self.fc_stroke_width1 = nn.Linear(5, 1)

#         self.wp, self.hp = canvas_width_pix, canvas_height_pix
#         self.wm, self.hm = canvas_width_meters, canvas_height_meters
#     def forward(self, real_params):
#         xs = (real_params[0,:4]/self.wm+.5) * self.wp
#         ys = (1-(real_params[0,4:8]/self.hm+.5)) * self.hp
#         z = real_params[0,8:]#torch.mean(real_params[0,8:]).view(1)#.unsqueeze(0)
#         # print(z.shape)
#         # z = self.fc_stroke_width(z)
#         z = self.fc_stroke_width1(nn.functional.relu(self.fc_stroke_width0(z)))
#         # print(z.shape, xs.shape)
#         return torch.cat([xs,ys,z]).unsqueeze(0)

# class WidthTranslation(nn.Module):
#     def __init__(self):
#         super(WidthTranslation, self).__init__()
#         self.a = nn.Parameter(torch.tensor(3.))
#         self.b = nn.Parameter(torch.tensor(4.))

#     def forward(self, real_widths):
#         return self.real_to_sim(real_widths)

#     def real_to_sim(self, real_widths):
#         return torch.mean(real_widths)*self.a + self.b
#     def sim_to_real(self, sim_width):
#         return (sim_width - self.b)/self.a

class WidthTranslation(nn.Module):
    def __init__(self):
        super(WidthTranslation, self).__init__()
        self.a = nn.Parameter(torch.tensor(3.))
        self.e = nn.Parameter(torch.tensor(1.))
        self.b = nn.Parameter(torch.tensor(4.))

    def forward(self, real_widths):
        return self.real_to_sim(real_widths)

    def real_to_sim(self, real_widths):
        return torch.mean(real_widths)**self.e*self.a + self.b
    def sim_to_real(self, sim_width):
        return ((sim_width - self.b)/self.a)**(1/self.e)

class SimToRealTranslation(nn.Module):
    def __init__(self, in_size=9, out_size=12):
        super(SimToRealTranslation, self).__init__()

        self.fc = nn.Linear(in_size, out_size)

    def forward(self, sim_params):
        return self.fc(sim_params)

if __name__ == '__main__':
    global opt, strokes_small, strokes_full
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    writer = TensorBoard('stroke_modelling_log/model_stroke_diffvg_{}'.format(run_name))
    opt.writer = writer

    strokes_small = load_brush_strokes(opt, scale_factor=5)
    strokes_full = load_brush_strokes(opt, scale_factor=1)

    real_stroke_params = []

    for stroke_ind in range(len(strokes_small)):
        stroke = strokes_full[stroke_ind]

        stroke_real_params = torch.tensor(all_strokes[stroke_ind]().trajectory)

        real_stroke = stroke.permute(2,0,1).unsqueeze(0).detach().to(device)

        writer.add_image('images/{}_real_stroke'.format(stroke_ind), np.clip(real_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)
        real_stroke_params.append(RealStrokeParameters(stroke_real_params[:,0], stroke_real_params[:,1], stroke_real_params[:,2]))

    real_stroke_params = [s.to(device) for s in real_stroke_params]

    h, w = strokes_full[0].shape[0], strokes_full[0].shape[1]

    # Let's train
    trans_model = WidthTranslation().to(device)
    optim = torch.optim.Adam(trans_model.parameters(), lr=1e-2)
    for it in range(3):
        run_loss = 0.
        for s_ind in range(len(strokes_small)):
            optim.zero_grad()
            real_z = real_stroke_params[s_ind].zs
            sim_z = trans_model(real_z)

            shape, shape_group = init_diffvg_brush_stroke(h, w)
            shape.points[:,0] = (real_stroke_params[s_ind].xs/opt.CANVAS_WIDTH+.5) * w
            shape.points[:,1] = (1-(real_stroke_params[s_ind].ys/opt.CANVAS_HEIGHT+.5)) * h

            shape.stroke_width = sim_z
            diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)

            loss = nn.L1Loss()(diffvg_stroke[0], strokes_full[s_ind].permute(2,0,1)[:3])
            loss.backward()
            optim.step()
            run_loss += loss.item()
        print(run_loss)
    print(trans_model.a, trans_model.b, trans_model.e)
    # Real to Sim no ML
    for val_ind in range(len(strokes_small)):
        with torch.no_grad():
            real_params = real_stroke_params[val_ind]
            h, w = strokes_full[val_ind].shape[0], strokes_full[val_ind].shape[1]

            shape, shape_group = init_diffvg_brush_stroke(h, w)
            shape.points[:,0] = (real_params.xs/opt.CANVAS_WIDTH+.5) * w
            shape.points[:,1] = (1-(real_params.ys/opt.CANVAS_HEIGHT+.5)) * h
            # print(shape.points)
            # print(real_params.xs)
            # print(real_params.ys)
            # shape.stroke_width = torch.tensor(4.0)
            z = trans_model.real_to_sim(real_params.zs)#torch.mean(real_params.zs)*3+4
            shape.stroke_width = z
            diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)
            # show_img(strokes_full[val_ind])
            # show_img(diffvg_stroke)
            writer.add_image('images_test_no_ml/{}_rel_stroke'.format(val_ind), np.clip(strokes_full[val_ind].permute(2,0,1).unsqueeze(0).detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)
            writer.add_image('images_test_no_ml/{}_sim_stroke'.format(val_ind), np.clip(diffvg_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)
    return trans_model
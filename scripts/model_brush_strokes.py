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

# def initialize_curves(num_paths, canvas_width, canvas_height):
#     shapes = []
#     shape_groups = []
#     for i in range(num_paths):
#         num_segments = random.randint(1, 1)
#         num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
#         points = []
#         p0 = (random.random(), random.random())
#         points.append(p0)
#         for j in range(num_segments):
#             radius = 0.1
#             p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
#             p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
#             p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
#             points.append(p1)
#             points.append(p2)
#             points.append(p3)
#             p0 = p3
#         points = torch.tensor(points)
#         points[:, 0] *= canvas_width
#         points[:, 1] *= canvas_height
#         path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor(1.0), is_closed = False)
#         shapes.append(path)
#         path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
#         shape_groups.append(path_group)
#     return shapes, shape_groups

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

def real_stroke_params_to_tensor(params):
    return params.flatten()

class RealStrokeParameters(nn.Module):
    def __init__(self, xs, ys, zs):
        super(RealStrokeParameters, self).__init__()

        self.xs = nn.Parameter(xs)
        self.ys = nn.Parameter(ys)
        self.zs = nn.Parameter(zs)
    def forward(self):
        return torch.cat([self.xs, self.ys, self.zs]).unsqueeze(0)

class DiffVGStrokeParameters(nn.Module):
    def __init__(self, xs, ys, width):
        super(DiffVGStrokeParameters, self).__init__()

        self.xs = nn.Parameter(xs)
        self.ys = nn.Parameter(ys)
        self.width = nn.Parameter(width)

    def forward(self):
        return torch.cat([self.xs, self.ys, self.width.view(1)]).unsqueeze(0)

# class RealToSimTranslation(nn.Module):
#     def __init__(self, in_size=12, out_size=9):
#         super(RealToSimTranslation, self).__init__()

#         self.fc = nn.Linear(in_size, out_size)

#     def forward(self, real_params):
#         return self.fc(real_params)

class RealToSimTranslation(nn.Module):
    def __init__(self, canvas_width_pix, canvas_height_pix, canvas_width_meters, canvas_height_meters):
        super(RealToSimTranslation, self).__init__()

        self.fc_stroke_width0 = nn.Linear(4, 5)
        self.fc_stroke_width1 = nn.Linear(5, 1)

        self.wp, self.hp = canvas_width_pix, canvas_height_pix
        self.wm, self.hm = canvas_width_meters, canvas_height_meters
    def forward(self, real_params):
        xs = (real_params[0,:4]/self.wm+.5) * self.wp
        ys = (1-(real_params[0,4:8]/self.hm+.5)) * self.hp
        z = real_params[0,8:]#torch.mean(real_params[0,8:]).view(1)#.unsqueeze(0)
        # print(z.shape)
        # z = self.fc_stroke_width(z)
        z = self.fc_stroke_width1(nn.functional.relu(self.fc_stroke_width0(z)))
        # print(z.shape, xs.shape)
        return torch.cat([xs,ys,z]).unsqueeze(0)

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
    sim_stroke_params = []

    if not os.path.exists('real_stroke_params.pt'):
        for stroke_ind in range(len(strokes_small)):

            stroke = strokes_full[stroke_ind]

            # print(stroke.shape)

            stroke_real_params = torch.tensor(all_strokes[stroke_ind]().trajectory)
            # print(stroke_real_params)

            h, w = stroke.shape[0], stroke.shape[1]
            with torch.no_grad():
                shape, shape_group = init_diffvg_brush_stroke(h, w)
                # print(shape.points, shape.stroke_width)
                # shape, shape_group = initialize_curves(1, w,h)
                # shape, shape_group = shape[0], shape_group[0]
                # print(shape.points, shape.stroke_width)

            real_stroke = stroke.permute(2,0,1).unsqueeze(0).detach().to(device)
            diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)

            writer.add_image('images/{}_real_stroke'.format(stroke_ind), np.clip(real_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)

            # print(real_stroke.shape, diffvg_stroke.shape)
            # print(real_stroke.max(), diffvg_stroke.max())
            # show_img(diffvg_stroke)
            # show_img(stroke)

            shape.points.requires_grad = True
            shape.stroke_width.requires_grad = True
            shape_group.stroke_color.requires_grad = True

            points = [shape.points]
            stroke_widths = [shape.stroke_width]
            colors = [shape_group.stroke_color]

            points_optim = torch.optim.RMSprop(points, lr=0.7)
            width_optim = torch.optim.RMSprop(stroke_widths, lr=0.3)
            color_optim = torch.optim.RMSprop(colors, lr=0.01)

            for it in range(70):
                points_optim.zero_grad()
                width_optim.zero_grad()
                color_optim.zero_grad()

                diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)

                # loss = torch.nn.L1Loss()(diffvg_stroke, real_stroke[:,:3])
                loss = torch.nn.L1Loss()(1-torch.mean(diffvg_stroke, dim=1), real_stroke[:,-1])

                #print(loss.item())
                loss.backward()

                points_optim.step()
                width_optim.step()
                # color_optim.step()

                if it % 10 ==0:
                    # show_img(diffvg_stroke.detach())
                    # show_img(real_stroke.detach())
                    writer.add_scalar('loss{}'.format(stroke_ind), loss.item(), it)
                    writer.add_image('images/{}_sim_stroke'.format(stroke_ind), np.clip(diffvg_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), it)
            real_stroke_params.append(RealStrokeParameters(stroke_real_params[:,0], stroke_real_params[:,1], stroke_real_params[:,2]))
            p = shape.points.detach().cpu()
            # print(p)
            sim_stroke_params.append(DiffVGStrokeParameters(p[:,0], p[:,1], shape.stroke_width.detach().cpu()))

            torch.save(real_stroke_params, 'real_stroke_params.pt')
            torch.save(sim_stroke_params, 'sim_stroke_params.pt')

    real_stroke_params = torch.load('real_stroke_params.pt')
    sim_stroke_params = torch.load('sim_stroke_params.pt')

    real_stroke_params = [s.to(device) for s in real_stroke_params]
    sim_stroke_params = [s.to(device) for s in sim_stroke_params]

    # sim_stroke_params = torch.cat([s.unsqueeze(0) for s in sim_stroke_params], dim=0)

    # Real to Sim no ML
    for val_ind in range(len(strokes_small)):
        with torch.no_grad():
            real_params = real_stroke_params[val_ind]
            h, w = strokes_full[val_ind].shape[0], strokes_full[val_ind].shape[1]
            CANVAS_WIDTH  = 0.3556 -0.005# 14"
            CANVAS_HEIGHT = 0.2794 -0.005# 11"

            shape, shape_group = init_diffvg_brush_stroke(h, w)
            shape.points[:,0] = (real_params.xs/CANVAS_WIDTH+.5) * w
            shape.points[:,1] = (1-(real_params.ys/CANVAS_HEIGHT+.5)) * h
            # print(shape.points)
            # print(real_params.xs)
            # print(real_params.ys)
            # shape.stroke_width = torch.tensor(4.0)
            z = torch.mean(real_params.zs)*3+4
            shape.stroke_width = z
            diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)
            # show_img(strokes_full[val_ind])
            # show_img(diffvg_stroke)
            writer.add_image('images_test_no_ml/{}_rel_stroke'.format(val_ind), np.clip(strokes_full[val_ind].permute(2,0,1).unsqueeze(0).detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)
            writer.add_image('images_test_no_ml/{}_sim_stroke'.format(val_ind), np.clip(diffvg_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)

    # # Sim to Real
    # for val_ind in range(len(strokes_small)):
    #     sim_to_real = SimToRealTranslation().to(device)

    #     optim = torch.optim.Adam(sim_to_real.parameters(), lr=1e-2)

    #     best_model, best_val_loss = None, 99999

    #     for it in range(4000):
    #         optim.zero_grad()
    #         loss = 0
    #         for stroke_ind in range(len(strokes_small)):
    #             if stroke_ind == val_ind:
    #                 continue
    #             loss += torch.nn.L1Loss()(real_stroke_params[stroke_ind]().detach(), sim_to_real(sim_stroke_params[stroke_ind]().detach()))
    #         loss.backward()
    #         optim.step()
    #         with torch.no_grad():
    #             val_loss = torch.nn.L1Loss()(real_stroke_params[val_ind]().detach(), sim_to_real(sim_stroke_params[val_ind]().detach()))

    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss 
    #                 best_model = copy.deepcopy(sim_to_real)

    #         if it % 10 ==0:
    #             writer.add_scalar('loss/{}_train_loss'.format(val_ind), loss.item(), it)
    #             writer.add_scalar('loss/{}_val_loss'.format(val_ind), val_loss.item(), it)

    #     pred_sim_stroke = best_model(sim_stroke_params[val_ind]().detach())
    #     h, w = strokes_full[val_ind].shape[0], strokes_full[val_ind].shape[1]
    #     with torch.no_grad():
    #         shape, shape_group = init_diffvg_brush_stroke(h, w)
    #         shape.points[:,0] = pred_sim_stroke[0,:4]
    #         shape.points[:,1] = pred_sim_stroke[0,4:8]
    #         shape.stroke_width = pred_sim_stroke[0,-1]
    #         diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)
    #         # show_img(strokes_full[val_ind])
    #         # show_img(diffvg_stroke)
    #         writer.add_image('images_test/{}_rel_stroke'.format(val_ind), np.clip(strokes_full[val_ind].permute(2,0,1).unsqueeze(0).detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), it)
    #         writer.add_image('images_test/{}_sim_stroke'.format(val_ind), np.clip(diffvg_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), it)














    # Real to Sim
    for val_ind in range(len(strokes_small)):
        # real_to_sim = RealToSimTranslation().to(device)
        h, w = strokes_full[val_ind].shape[0], strokes_full[val_ind].shape[1]
        real_to_sim = RealToSimTranslation(canvas_width_pix=w, canvas_height_pix=h, canvas_width_meters=0.3556, canvas_height_meters=0.2794).to(device)

        optim = torch.optim.Adam(real_to_sim.parameters(), lr=.5)

        best_model, best_val_loss = None, 99999

        for it in range(200):
            optim.zero_grad()
            loss = 0
            for stroke_ind in range(len(strokes_small)):
                if stroke_ind == val_ind:
                    continue
                loss += torch.nn.L1Loss()(sim_stroke_params[stroke_ind]().detach(), real_to_sim(real_stroke_params[stroke_ind]().detach()))
            loss.backward()
            optim.step()
            with torch.no_grad():
                val_loss = torch.nn.L1Loss()(sim_stroke_params[val_ind]().detach(), real_to_sim(real_stroke_params[val_ind]().detach()))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss 
                    best_model = copy.deepcopy(real_to_sim)

            if it % 10 ==0:
                writer.add_scalar('loss/{}_train_loss'.format(val_ind), loss.item(), it)
                writer.add_scalar('loss/{}_val_loss'.format(val_ind), val_loss.item(), it)

        pred_sim_stroke = best_model(real_stroke_params[val_ind]().detach())
        h, w = strokes_full[val_ind].shape[0], strokes_full[val_ind].shape[1]
        with torch.no_grad():
            shape, shape_group = init_diffvg_brush_stroke(h, w)
            shape.points[:,0] = pred_sim_stroke[0,:4]
            shape.points[:,1] = pred_sim_stroke[0,4:8]
            shape.stroke_width = pred_sim_stroke[0,-1]
            diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)
            # show_img(strokes_full[val_ind])
            # show_img(diffvg_stroke)
            writer.add_image('images_test/{}_rel_stroke'.format(val_ind), np.clip(strokes_full[val_ind].permute(2,0,1).unsqueeze(0).detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), it)
            writer.add_image('images_test/{}_sim_stroke'.format(val_ind), np.clip(diffvg_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), it)

        # pred_sim_stroke = best_model(real_stroke_params[val_ind+1]().detach())
        # with torch.no_grad():
        #     shape, shape_group = init_diffvg_brush_stroke(h, w)
        #     shape.points[:,0] = pred_sim_stroke[0,:4]
        #     shape.points[:,1] = pred_sim_stroke[0,4:8]
        #     shape.stroke_width = pred_sim_stroke[0,-1]
        #     diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)
        #     show_img(strokes_full[val_ind+1])
        #     show_img(diffvg_stroke)

        avg_train_loss = 0
        with torch.no_grad():
            for stroke_ind in range(len(strokes_small)):
                if stroke_ind == val_ind:
                    continue
                avg_train_loss += torch.nn.L1Loss()(sim_stroke_params[stroke_ind]().detach(), best_model(real_stroke_params[stroke_ind]().detach()))
            avg_train_loss /= len(strokes_small)-1
        print(avg_train_loss, best_val_loss)

    # Sim 2 Real
    for val_ind in range(len(strokes_small)):
        sim_to_real = SimToRealTranslation().to(device)

        optim = torch.optim.Adam(sim_to_real.parameters(), lr=1e-2)

        for it in range(200):
            optim.zero_grad()
            loss = 0
            for stroke_ind in range(len(strokes_small)):
                if stroke_ind == val_ind:
                    continue
                loss += torch.nn.L1Loss()(real_stroke_params[stroke_ind]().detach(), sim_to_real(sim_stroke_params[stroke_ind]().detach()))
            loss.backward()
            optim.step()
            with torch.no_grad():
                val_loss = torch.nn.L1Loss()(real_stroke_params[val_ind]().detach(), sim_to_real(sim_stroke_params[val_ind]().detach()))

            if it % 10 ==0:
                writer.add_scalar('loss/{}_train_loss'.format(val_ind), loss.item(), it)
                writer.add_scalar('loss/{}_val_loss'.format(val_ind), val_loss.item(), it)
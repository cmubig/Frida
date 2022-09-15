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
import kornia as K
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

# import pydiffvg

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

# class RealStrokeParameters(nn.Module):
#     def __init__(self, xs, ys, zs):
#         super(RealStrokeParameters, self).__init__()

#         self.xs = nn.Parameter(xs)
#         self.ys = nn.Parameter(ys)
#         self.zs = nn.Parameter(zs)
#     def forward(self):
#         return torch.cat([self.xs, self.ys, self.zs]).unsqueeze(0)
# class RealStrokeParameters(nn.Module):
#     def __init__(self, xs, ys, zs):
#         super(RealStrokeParameters, self).__init__()

#         self.xs = nn.Parameter(xs)
#         self.ys = nn.Parameter(ys)
#         self.zs = nn.Parameter(zs)
#     def forward(self):
#         return torch.cat([self.xs, self.ys, self.zs]).unsqueeze(0)

# class StrokeParametersToImage(nn.Module):
#     def __init__(self, h, w):
#         super(StrokeParametersToImage, self).__init__()
#         self.fc1 = nn.Linear(12, w*h)
#         # self.fc1 = nn.Linear(12, 20)
#         # self.fc2 = nn.Linear(20, w*h)
#         self.w = w 
#         self.h = h 

#     def forward(self, x):
#         return self.fc1(x).view(-1, h, w)
#         # return self.fc2(self.fc1(x)).view(-1, h, w)

class StrokeParametersToImage(nn.Module):
    def __init__(self, h, w):
        super(StrokeParametersToImage, self).__init__()
        # self.fc1 = nn.Linear(12, w*h)
        self.fc1 = nn.Linear(12, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, w*h)
        self.w = w 
        self.h = h 

    def forward(self, x):
        # return self.fc1(x).view(-1, h, w)
        return self.fc4(self.fc3(self.fc2(self.fc1(x)))).view(-1, h, w)

if __name__ == '__main__':
    global opt, strokes_small, strokes_full
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    writer = TensorBoard('painting_stroke_modelling/model_stroke{}'.format(run_name))
    opt.writer = writer

    strokes_small = load_brush_strokes(opt, scale_factor=5)
    strokes_full = load_brush_strokes(opt, scale_factor=1)

    strokes_small = strokes_full

    h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]

    for stroke_ind in range(len(strokes_small)):
        # Crop
        strokes_small[stroke_ind] = strokes_small[stroke_ind][int(.4*h):int(0.6*h), int(0.4*w):int(0.8*w),:]

        # Media filter
        strokes_small[stroke_ind] = K.filters.median_blur(
            strokes_small[stroke_ind].permute(2,0,1).unsqueeze(0), 
            kernel_size=(11,11))[0].permute(1,2,0)
    h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]
    print(len(strokes_small), strokes_small[0].shape, strokes_small[0].max())

    

    trans = StrokeParametersToImage(h,w).to(device)

    val_stroke_ind = 7

    for stroke_ind in range(len(strokes_small)):
        writer.add_image('images/{}_real_stroke'.format(stroke_ind), 
            np.clip(strokes_small[stroke_ind][:,:,-1].detach().cpu().numpy(), a_min=0, a_max=1)*255, 0)




    

    for val_ind in range(len(strokes_small)):
        trans = StrokeParametersToImage(h,w).to(device)
        optim = torch.optim.Adam(trans.parameters(), lr=1e-2)
        best_model = None
        best_val_loss = 999

        for it in range(700):
            ep_loss = 0.
            # for stroke_ind in range(len(strokes_small)):
            #     if stroke_ind == val_ind:
            #         continue

            #     optim.zero_grad()

            #     stroke_real_params = all_strokes[stroke_ind]().trajectory
            #     x = torch.tensor(np.array(stroke_real_params).flatten()).float().to(device)
            #     pred_stroke = trans(x)

            #     real_stroke = strokes_small[stroke_ind][:,:,-1].unsqueeze(0)
            #     loss = nn.L1Loss()(pred_stroke, real_stroke)

            #     loss.backward()
            #     optim.step()
            #     ep_loss += loss.item()

            xs = []
            real_strokes = []
            for stroke_ind in range(len(strokes_small)):
                if stroke_ind == val_ind:
                    continue

                stroke_real_params = all_strokes[stroke_ind]().trajectory
                x = torch.tensor(np.array(stroke_real_params).flatten()).float().to(device).unsqueeze(0)
                xs.append(x)

                real_stroke = strokes_small[stroke_ind][:,:,-1].unsqueeze(0)
                real_strokes.append(real_stroke)
            xs = torch.cat(xs, dim=0)

            real_strokes = torch.cat(real_strokes, dim=0)
                
            optim.zero_grad()
            pred_strokes = trans(xs)

            # loss = nn.L1Loss()(pred_strokes, real_strokes)
            loss = nn.MSELoss()(pred_strokes, real_strokes)

            ep_loss = loss.item()
            loss.backward()
            optim.step()
            
            with torch.no_grad():
                stroke_real_params = all_strokes[val_ind]().trajectory
                x = torch.tensor(np.array(stroke_real_params).flatten()).float().to(device)
                pred_stroke = trans(x)
                if it % 50 == 0:
                    writer.add_image('images/{}_sim_stroke'.format(val_ind), 
                        np.clip(pred_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)*255, it)

                real_stroke = strokes_small[val_ind][:,:,-1].unsqueeze(0)
                loss = nn.L1Loss()(pred_stroke, real_stroke)
                writer.add_scalar('loss/val_loss_{}'.format(val_ind), loss.item(), it)

                if loss.item() < best_val_loss and it > 50:
                    best_val_loss = loss.item()
                    best_model = copy.deepcopy(trans)
        with torch.no_grad():
            stroke_real_params = all_strokes[val_ind]().trajectory
            x = torch.tensor(np.array(stroke_real_params).flatten()).float().to(device)
            pred_stroke = best_model(x)
            writer.add_image('images/{}_sim_stroke'.format(val_ind), 
                    np.clip(pred_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)*255, 1000)


    1/0 
    ############
    optim = torch.optim.Adam(trans.parameters(), lr=1e-2)

    for it in range(300):
        ep_loss = 0.
        for stroke_ind in range(len(strokes_small)):
            optim.zero_grad()

            stroke_real_params = all_strokes[stroke_ind]().trajectory
            x = torch.tensor(np.array(stroke_real_params).flatten()).float().to(device)
            #print('x', x)
            pred_stroke = trans(x)
            #print('stroke predict size', pred_stroke.shape, strokes_small[0].shape)

            real_stroke = strokes_small[stroke_ind][:,:,-1].unsqueeze(0)
            loss = nn.L1Loss()(pred_stroke, real_stroke)

            loss.backward()
            optim.step()
            ep_loss += loss.item()

            if it % 50 == 0:
                writer.add_image('images/{}_sim_stroke'.format(stroke_ind), 
                    np.clip(pred_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)*255, it)
        #print(ep_loss)
        writer.add_scalar('loss/train_loss', ep_loss, it)

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

# import pydiffvg

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')


if __name__ == '__main__':
    global opt, strokes_small, strokes_full
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    writer = TensorBoard('painting/model_stroke{}'.format(run_name))
    opt.writer = writer

    strokes_small = load_brush_strokes(opt, scale_factor=5)
    strokes_full = load_brush_strokes(opt, scale_factor=1)

    stroke_ind = 7

    stroke = strokes_full[stroke_ind]


    

    print(stroke.shape)

    stroke_real_params = all_strokes[stroke_ind]().trajectory
    print(stroke_real_params)

    h, w = stroke.shape[0], stroke.shape[1]
    

    real_stroke = stroke.permute(2,0,1).unsqueeze(0).detach().to(device)
    stroke_stamp = strokes_full[1].permute(2,0,1).unsqueeze(0).detach().to(device)

    writer.add_image('images/real_stroke', np.clip(real_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)
    
    B,C,H,W = real_stroke.size()
    # mesh grid
    xx = torch.arange(0,W).view(1,-1).repeat(H,1)
    yy = torch.arange(0,H).view(-1,1).repeat(1,W)

    xx = xx.view(1,H,W,1).repeat(B,1,1,1)
    yy = yy.view(1,H,W,1).repeat(B,1,1,1)

    grid = torch.cat((xx,yy),3).float()
    flow = grid.to(device)
        
    # scale grid to [-1,1]
    flow[:,:,:,0] = 2.0*flow[:,:,:,0].clone()/max(W-1,1)-1.0
    flow[:,:,:,1] = 2.0*flow[:,:,:,1].clone()/max(H-1,1)-1.0

    flow.requires_grad = True
    optim = torch.optim.Adam([flow], lr=1e-2)

    for it in range(20000):
        optim.zero_grad()

        sim_stroke = torch.nn.functional.grid_sample(stroke_stamp.detach(),flow)
        # print(sim_stroke.shape, real_stroke.shape)

        loss = torch.nn.L1Loss()(sim_stroke[:,-1:], real_stroke[:,-1:])

        # print(loss.item())
        loss.backward()

        optim.step()

        if it % 100 ==0:
            # show_img(sim_stroke.detach())
            # show_img(real_stroke.detach())
            writer.add_image('images/sim_stroke', np.clip(sim_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), it)
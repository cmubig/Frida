

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

from strokes import all_strokes



from plan_all_strokes import load_brush_strokes, show_img

from diffvg_helper import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')


class RealStrokeParameters(nn.Module):
    def __init__(self, xs, ys, zs):
        super(RealStrokeParameters, self).__init__()

        self.xs = nn.Parameter(xs)
        self.ys = nn.Parameter(ys)
        self.zs = nn.Parameter(zs)
    def forward(self):
        return torch.cat([self.xs, self.ys, self.zs]).unsqueeze(0)


def get_stroke_width_translation_model():
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

        # real_stroke = stroke.permute(2,0,1).unsqueeze(0).detach().to(device)

        # writer.add_image('images/{}_real_stroke'.format(stroke_ind), np.clip(real_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)
        real_stroke_params.append(RealStrokeParameters(stroke_real_params[:,0], 
            stroke_real_params[:,1], stroke_real_params[:,2]))

    real_stroke_params = [s.to(device) for s in real_stroke_params]

    h, w = strokes_full[0].shape[0], strokes_full[0].shape[1]

    # Let's train
    trans_model = WidthTranslation().to(device)
    trans_model.requires_grad = True
    optim = torch.optim.Adam(trans_model.parameters(), lr=1e-4)

    for it in range(0 if opt.simulate else 1):
        run_loss = 0.
        for s_ind in range(len(strokes_small)):
            optim.zero_grad()
            real_z = real_stroke_params[s_ind].zs
            #print(h)
            sim_z = trans_model(real_z)*h
            # print(sim_z)

            shape, shape_group = init_diffvg_brush_stroke(h, w)
            shape.points[:,0] = (real_stroke_params[s_ind].xs/opt.CANVAS_WIDTH+.5) * w
            shape.points[:,1] = (1-(real_stroke_params[s_ind].ys/opt.CANVAS_HEIGHT+.5)) * h

            shape.stroke_width = sim_z
            diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)

            # loss = nn.L1Loss()(diffvg_stroke[0], strokes_full[s_ind].permute(2,0,1)[:3])
            # loss = nn.L1Loss()(diffvg_stroke[0].sum(), strokes_full[s_ind].permute(2,0,1)[:3])
            p = (diffvg_stroke[0] / diffvg_stroke.max()).mean(dim=0)

            t = strokes_full[s_ind].permute(2,0,1)[3:] #/ strokes_full[s_ind].permute(2,0,1)[3:].max()
            t[t>0.3] = 1.
            # if s_ind == 0: show_img(t), show_img(1-p)
            # print(p.sum(), (1-t).sum())
            loss = nn.L1Loss()((1-p).sum(), (t).sum())
            # loss.backward()
            # optim.step()
            run_loss += loss#.item()

            # del loss, diffvg_stroke, shape, shape_group
        run_loss.backward()
        optim.step()
        print(run_loss)
    # print(trans_model.a, trans_model.b, trans_model.e)
    print(trans_model.a, trans_model.b)
    # Real to Sim no ML

    max_sim_z = -1
    min_sim_z = 99999
    max_stroke_length = -1
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
            shape.stroke_width = z*h
            diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)
            # # show_img(strokes_full[val_ind])
            # # show_img(diffvg_stroke)
            writer.add_image('images_test_no_ml/{}_rel_stroke'.format(val_ind), np.clip(strokes_full[val_ind].permute(2,0,1).unsqueeze(0).detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)
            writer.add_image('images_test_no_ml/{}_sim_stroke'.format(val_ind), np.clip(diffvg_stroke.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), 0)

            if z > max_sim_z: max_sim_z = float(z.detach())
            if z < min_sim_z: min_sim_z = float(z.detach())

            xs, xe = shape.points[0,0], shape.points[-1,0]
            ys, ye = shape.points[0,1], shape.points[-1,1]
            stroke_length = ((xe-xs)**2 + (ye-ys)**2)**(.5)/h
            if stroke_length > max_stroke_length: max_stroke_length = stroke_length
            # print('\n', val_ind)
            # print(z)
            est_real_z = trans_model.sim_to_real(z)
            # print(est_real_z)
    # print('min max sim z',  min_sim_z, max_sim_z)
    trans_model.eval()
    trans_model.requires_grad=False
    torch.cuda.empty_cache()
    return trans_model, min_sim_z, max_sim_z, max_stroke_length


if __name__ == '__main__':
    trans_model, min_sim_z, max_sim_z, max_stroke_length= get_stroke_width_translation_model()


    opt = Options()
    opt.gather_options()

    with open(os.path.join(opt.cache_dir, 'min_sim_z.pkl'),'wb') as f:
        pickle.dump(min_sim_z, f)
    with open(os.path.join(opt.cache_dir, 'max_sim_z.pkl'),'wb') as f:
        pickle.dump(max_sim_z, f)
    with open(os.path.join(opt.cache_dir, 'max_stroke_length.pkl'),'wb') as f:
        pickle.dump(max_stroke_length, f)
    torch.save(trans_model.state_dict(), os.path.join(opt.cache_dir, 'trans_model.pkl'))
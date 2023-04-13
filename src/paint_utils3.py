
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import numpy as np
import torch
from torchvision import transforms
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import colour
import random
# import gzip

# from torch_painting_models_continuous_concerted import *
from torch_painting_models_continuous import *
from clip_attn.clip_attn import get_attention

from my_tensorboard import TensorBoard

def load_img(path, h=None, w=None):
    im = Image.open(path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im = np.array(im)
    # if im.shape[1] > max_size:
    #     fact = im.shape[1] / max_size
    im = cv2.resize(im, (w,h)) if h is not None and w is not None else im
    im = torch.from_numpy(im)
    im = im.permute(2,0,1)
    return im.unsqueeze(0).float()

def get_colors(img, n_colors=6):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(img.reshape((img.shape[0]*img.shape[1],3)))
    colors = [kmeans.cluster_centers_[i] for i in range(len(kmeans.cluster_centers_))]
    colors = (torch.from_numpy(np.array(colors)) / 255.).float().to(device)
    return colors

def show_img(img, display_actual_size=True):
    if type(img) is torch.Tensor:
        img = img.detach().cpu().numpy()

    img = img.squeeze()
    if img.shape[0] < 5:
        img = img.transpose(1,2,0)

    if img.max() > 4:
        img = img / 255.
    img = np.clip(img, a_min=0, a_max=1)

    if display_actual_size:
        # Display at actual size: https://stackoverflow.com/questions/60144693/show-image-in-its-original-resolution-in-jupyter-notebook
        # Acquire default dots per inch value of matplotlib
        dpi = matplotlib.rcParams['figure.dpi']
        # Determine the figures size in inches to fit your image
        height, width = img.shape[0], img.shape[1]
        figsize = width / float(dpi), height / float(dpi)

        plt.figure(figsize=figsize)

    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    #plt.scatter(img.shape[1]/2, img.shape[0]/2)
    plt.show()


def sort_brush_strokes_by_color(painting, bin_size=3000):
    with torch.no_grad():
        brush_strokes = [bs for bs in painting.brush_strokes]
        for j in range(0,len(brush_strokes), bin_size):
            brush_strokes[j:j+bin_size] = sorted(brush_strokes[j:j+bin_size], 
                key=lambda x : x.color_transform.mean()+x.color_transform.prod(), 
                reverse=True)
        painting.brush_strokes = nn.ModuleList(brush_strokes)
        return painting

def randomize_brush_stroke_order(painting):
    with torch.no_grad():
        brush_strokes = [bs for bs in painting.brush_strokes]
        random.shuffle(brush_strokes)
        painting.brush_strokes = nn.ModuleList(brush_strokes)
        return painting

def discretize_colors(painting, discrete_colors):
    # pass
    with torch.no_grad():
        for brush_stroke in painting.brush_strokes:
            new_color = discretize_color(brush_stroke, discrete_colors)
            brush_stroke.color_transform.data *= 0
            brush_stroke.color_transform.data += new_color

def discretize_color(brush_stroke, discrete_colors):
    dc = discrete_colors.cpu().detach().numpy()
    #print('dc', dc.shape)
    dc = dc[None,:,:]
    #print('dc', dc.shape, dc.max())
    dc = cv2.cvtColor(dc, cv2.COLOR_RGB2Lab)
    #print('dc', dc.shape, dc.max())
    with torch.no_grad():
        color = brush_stroke.color_transform.detach()
        # dist = torch.mean(torch.abs(discrete_colors - color[None,:])**2, dim=1)
        # argmin = torch.argmin(dist)
        c = color[None,None,:].detach().cpu().numpy()
        #print('c', c.shape)
        # print(c)
        c = cv2.cvtColor(c, cv2.COLOR_RGB2Lab)
        #print('c', c.shape)

        
        dist = colour.delta_E(dc, c)
        #print(dist.shape)
        argmin = np.argmin(dist)

        return discrete_colors[argmin].clone()


def save_colors(allowed_colors):
    """
    Save the colors used as an image so you know how to mix the paints
    args:
        allowed_colors tensor
    """
    n_colors = len(allowed_colors)
    i = 0
    w = 128
    big_img = np.ones((2*w, 6*w, 3))

    for c in allowed_colors:
        c = allowed_colors[i].cpu().detach().numpy()
        c = c[::-1]
        big_img[(i//6)*w:(i//6)*w+w, (i%6)*w:(i%6)*w+w,:] = np.concatenate((np.ones((w,w,1))*c[2], np.ones((w,w,1))*c[1], np.ones((w,w,1))*c[0]), axis=-1)
        
        i += 1
    while i < 12:
        big_img[(i//6)*w:(i//6)*w+w, (i%6)*w:(i%6)*w+w,:] = np.concatenate((np.ones((w,w,1)), np.ones((w,w,1)), np.ones((w,w,1))), axis=-1)
        i += 1

    return big_img

def save_painting_strokes(painting, opt):
    # brush_stroke = BrushStroke(random.choice(strokes_small)).to(device)
    canvas = transforms.Resize(size=(h,w))(painting.background_img)

    individual_strokes = torch.empty((len(painting.brush_strokes),canvas.shape[1], canvas.shape[2], canvas.shape[3]))
    running_strokes = torch.empty((len(painting.brush_strokes),canvas.shape[1], canvas.shape[2], canvas.shape[3]))

    with torch.no_grad():
        for i in range(len(painting.brush_strokes)):                
            single_stroke = painting.brush_strokes[i](h,w)

            canvas = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
            
            running_strokes[i] = canvas 
            individual_strokes[i] = single_stroke

    running_strokes = (running_strokes.detach().cpu().numpy().transpose(0,2,3,1)*255).astype(np.uint8)
    individual_strokes = (individual_strokes.detach().cpu().numpy().transpose(0,2,3,1)*255).astype(np.uint8)

    with open(os.path.join(painter.opt.cache_dir, 'running_strokes.npy'), 'wb') as f:
        np.save(f, running_strokes)
    with open(os.path.join(painter.opt.cache_dir, 'individual_strokes.npy'), 'wb') as f:
        np.save(f, individual_strokes)

    for i in range(len(running_strokes)):
        if i % 5 == 0 or i == len(running_strokes)-1:
            opt.writer.add_image('images/plan', running_strokes[i], i)
    for i in range(len(individual_strokes)):
        if i % 5 == 0 or i == len(running_strokes)-1:
            opt.writer.add_image('images/individual_strokes', individual_strokes[i], i)



def random_init_painting(background_img, n_strokes, ink=False):
    gridded_brush_strokes = []
    xys = [(x,y) for x in torch.linspace(-.95,.95,int(n_strokes**0.5)) \
                 for y in torch.linspace(-.95,.95,int(n_strokes**0.5))]
    random.shuffle(xys)
    for x,y in xys:
        # Random brush stroke
        brush_stroke = BrushStroke(xt=x, yt=y, ink=ink)
        gridded_brush_strokes.append(brush_stroke)

    painting = Painting(0, background_img=background_img, 
        brush_strokes=gridded_brush_strokes).to(device)
    return painting


def create_tensorboard(log_dir='painting'):
    def new_tb_entry():
        import datetime
        date_and_time = datetime.datetime.now()
        run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
        return '{}/{}_planner'.format(log_dir, run_name)
    try:
        if IN_COLAB:
            tensorboard_dir = new_tb_entry()
        else:
            b = log_dir
            all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
            tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
            if '_planner' not in tensorboard_dir:
                tensorboard_dir += '_planner'
    except:
        tensorboard_dir = new_tb_entry()
    tensorboard_dir = new_tb_entry()
    return TensorBoard(tensorboard_dir)



def parse_csv_line_continuous(line):
    toks = line.split(',')
    if len(toks) != 9:
        return None
    x = float(toks[0])
    y = float(toks[1])
    r = float(toks[2])
    length = float(toks[3])
    thickness = float(toks[4])
    bend = float(toks[5])
    color = np.array([float(toks[6]), float(toks[7]), float(toks[8])])


    return x, y, r, length, thickness, bend, color

def format_img(tensor_img):
    np_painting = tensor_img.detach().cpu().numpy()[0].transpose(1,2,0)
    if np_painting.shape[-1] == 1:
        np_painting = cv2.cvtColor(np.float32(np_painting), cv2.COLOR_GRAY2RGB)
    return np.clip(np_painting, a_min=0, a_max=1)


def init_brush_strokes(attn, n_strokes, ink):
    brush_strokes = []
    
    attn += 0.05
    prob = attn / attn.sum()
    prob = prob.flatten()
    
    ravelled_inds = np.random.choice(len(prob), size=n_strokes, p=prob)
    xys = np.array(np.unravel_index(ravelled_inds, attn.shape))
                   
    for i in range(xys.shape[1]):
        x, y = xys[1,i]/attn.shape[1]*2-1, xys[0,i]/attn.shape[0]*2-1
        # Random brush stroke
        brush_stroke = BrushStroke(xt=x, yt=y, ink=ink)
        brush_strokes.append(brush_stroke)
    return brush_strokes

def initialize_painting(n_strokes, target_img, background_img, ink):
    attn = get_attention(target_img)
    brush_strokes = init_brush_strokes(attn, n_strokes, ink)
    painting = Painting(0, background_img=background_img, 
        brush_strokes=brush_strokes).to(device)
    return painting

def add_strokes_to_painting(painting, n_strokes, target_img, ink):
    attn = get_attention(target_img)
    brush_strokes = init_brush_strokes(attn, n_strokes, ink)
    painting = Painting(0, background_img=painting.background_img, 
        brush_strokes=painting.brush_strokes+brush_strokes).to(device)
    return painting

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import numpy as np
import torch
from torch import nn
import matplotlib
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False
  
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import colour
import random

from painting import Painting, BrushStroke
from clip_attn.clip_attn import get_attention

from my_tensorboard import TensorBoard


def canvas_to_global_coordinates(x,y,z, opt):
    ''' Canvas coordinates are proportions from the bottom left of canvas 
        Global coordinates are in meters wrt to the robot's center
    '''
    x_new = (x -.5) * opt.CANVAS_WIDTH + opt.CANVAS_POSITION[0]
    y_new = y*opt.CANVAS_HEIGHT + opt.CANVAS_POSITION[1]
    z_new = z
    return x_new, y_new, z_new

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

def to_video(frames, fn='animation{}.mp4'.format(time.time()), frame_rate=10):
    if len(frames) == 0: return
    h, w = frames[0].shape[0], frames[0].shape[1]
    # print(h,w)
    _fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # _fourcc = cv2.VideoWriter_fourcc(*'H264')
    # _fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(fn, _fourcc, frame_rate, (w,h))
    for frame in frames:
        cv2_frame = np.clip(frame, a_min=0, a_max=1) if frame.max() < 2 else frame / 255.
        cv2_frame = (cv2_frame * 255.).astype(np.uint8)[:,:,::-1]
        out.write(cv2_frame)
    out.release()

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

def sort_brush_strokes_by_location(painting, bin_size=3000):
    from scipy.spatial import distance_matrix
    points = np.zeros((2, len(painting.brush_strokes)))
    for i in range(len(painting.brush_strokes)):
        points[0,i] = painting.brush_strokes[i].transformation.xt.detach().cpu().numpy()
        points[1,i] = painting.brush_strokes[i].transformation.yt.detach().cpu().numpy()
    d_mat = distance_matrix(points.T, points.T)
    
    from tsp_solver.greedy import solve_tsp
    ordered_stroke_inds = solve_tsp(d_mat)

    with torch.no_grad():
        brush_strokes = [painting.brush_strokes[i] for i in ordered_stroke_inds]
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

def rgb2lab(image_rgb):
    image_rgb = image_rgb.astype(np.float32)
    if image_rgb.max() > 2:
        image_rgb = image_rgb / 255.

    # Also Consider
    # image_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(image_rgb))

    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    return image_lab

def lab2rgb(image_lab):
    image_rgb = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
    return image_rgb

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


def nearest_color(color, discrete_colors):
    ''' Get the most similar color to a given color (np.array([3])) '''
    #dist = np.mean(np.abs(discrete_colors - color[None,:])**2, axis=1)
    #argmin = np.argmin(dist)
    diffs = [compare_images(rgb2lab(c[np.newaxis, np.newaxis]), rgb2lab(color[np.newaxis, np.newaxis])) for c in discrete_colors]
    color_ind = np.argmin(np.array(diffs))
    return color_ind, discrete_colors[color_ind]

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


def extract_paint_color(canvas_before, canvas_after, stroke_bool_map):
    ''' Given a picture of the canvas before and after
    a brush stroke, extract the rgb color '''

    # Get a boolean map of pixels that changed significantly from the two photos
    # stroke_bool_map = cv2.resize(stroke_bool_map, (canvas_before.shape[1], canvas_before.shape[0]))
    # stroke_bool_map = stroke_bool_map > 0.3


    # Median filter target image so that it's not detailed
    og_shape = canvas_before.shape
    canvas_before_ = cv2.resize(canvas_before, (256,256)) # For scipy memory error
    canvas_after_ = cv2.resize(canvas_after, (256,256)) # For scipy memory error

    diff = np.max(np.abs(canvas_after_.astype(np.float32) - canvas_before_.astype(np.float32)), axis=2)
    diff = diff / 255.#diff.max()

    # smooth the diff
    diff = median_filter(diff, size=(5,5))
    diff = cv2.resize(diff,  (og_shape[1], og_shape[0]))

    stroke_bool_map = diff > .3
    if stroke_bool_map.astype(np.float32).sum() < 10: # at least 10 pixels
        return None

    # ca = canvas_after.copy()
    # ca[stroke_bool_map] = 0
    # show_img(ca)

    color = [np.median(canvas_after[:,:,ch][stroke_bool_map]) for ch in range(3)]


    # ca = canvas_after.copy()
    # print(color)
    # # for i in range(3):
    # #     ca[stroke_bool_map][i] = 0
    # # show_img(ca)
    # ca[stroke_bool_map] = np.array(color) 
    # show_img(ca)
    return np.array(color)

def random_init_painting(background_img, n_strokes, ink=False, device='cuda'):
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


def init_brush_strokes(diff, n_strokes, ink):
    if not IN_COLAB: matplotlib.use('TkAgg')
    brush_strokes = []
    
    if ink:
        diff[diff < 0.15] = 0
    else:
        diff[diff < 0.15] = 0.0001
    diff = diff.cpu().detach().numpy()
    points = (np.array(np.nonzero(diff))).astype(int)


    # from scipy.spatial import distance_matrix

    # #for j in range(8):
    # # while(points.shape[1] > n_strokes*1.5):
    # for j in range(2):
    #     d_mat = distance_matrix(points.T, points.T)
    #     for i in range(len(d_mat)):
    #         d_mat[i,i] = 1e9

    #     min_dists = d_mat.argmin(axis=0)
    #     # print('min_dists', min_dists.shape, min_dists[:50])
    #     min_inds = np.maximum(np.arange(points.shape[1]), min_dists)
    #     point_inds = np.unique(min_inds)
    #     points = points[:,point_inds]
    #     print('points', points.shape, point_inds.shape)

    # Subsample
    if n_strokes > 0:
        sub_samp = list(np.arange(points.shape[1]))
        import random 
        random.shuffle(sub_samp)
        points = points[:,np.array(sub_samp[:(min(n_strokes, points.shape[1]))])]
        # print('points', points.shape, points.max(), points[:,0])
    else:
        points = points[:,:0]
                   
    for i in range(points.shape[1]):
        x, y = points[1,i]/diff.shape[1]*2-1, points[0,i]/diff.shape[0]*2-1
        # Random brush stroke
        brush_stroke = BrushStroke(xt=x, yt=y, ink=ink, 
                                   stroke_length=torch.Tensor([0.001]))
        brush_strokes.append(brush_stroke)
    return brush_strokes

def initialize_painting(n_strokes, target_img, background_img, ink, device='cuda'):
    attn = (target_img[0] - background_img[0]).abs().mean(dim=0)
    brush_strokes = init_brush_strokes(attn, n_strokes, ink)
    painting = Painting(0, background_img=background_img, 
        brush_strokes=brush_strokes).to(device)
    return painting

def add_strokes_to_painting(painting, rendered_painting, n_strokes, target_img, background_img, ink, device='cuda'):
    attn = (target_img[0] - rendered_painting[0]).abs().mean(dim=0)
    brush_strokes = init_brush_strokes(attn, n_strokes, ink)
    existing_strokes = [painting.brush_strokes[i] for i in range(len(painting.brush_strokes))]
    painting = Painting(0, background_img=background_img, 
        brush_strokes=existing_strokes+brush_strokes).to(device)
    return painting

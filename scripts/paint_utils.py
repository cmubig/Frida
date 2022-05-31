#! /usr/bin/env python3

##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import cv2 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
from scipy.ndimage import median_filter
import time
import os
import pickle

import colour

def rgb2lab(image_rgb):
    image_rgb = image_rgb.astype(np.float32)
    if image_rgb.max() > 2:
        image_rgb = image_rgb / 255.

    # Also Consider
    # image_lab = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(image_rgb))

    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    return image_lab

def compare_images(img1, img2):
    ''' Pixel wise comparison '''
    # Input images are Lab
    delta_E = colour.delta_E(img1, img2)
    return delta_E

def canvas_to_global_coordinates(x,y,z, opt):
    ''' Canvas coordinates are proportions from the bottom left of canvas 
        Global coordinates are in meters wrt to the robot's center
    '''
    x_new = (x -.5) * opt.CANVAS_WIDTH + opt.CANVAS_POSITION[0]
    y_new = y*opt.CANVAS_HEIGHT + opt.CANVAS_POSITION[1]
    z_new = z
    return x_new, y_new, z_new

# def global_to_canvas_coordinates(x,y,z):
#     x_new = x + self.opt.CANVAS_POSITION[0]/2
#     y_new = y - self.opt.CANVAS_POSITION[1]
#     z_new = z
#     return x_new, y_new, z_new

def show_img(img, title=''):
    # Display at actual size: https://stackoverflow.com/questions/60144693/show-image-in-its-original-resolution-in-jupyter-notebook
    # Acquire default dots per inch value of matplotlib
    dpi = matplotlib.rcParams['figure.dpi']
    # Determine the figures size in inches to fit your image
    height, width = img.shape[0], img.shape[1]
    figsize = width / float(dpi), height / float(dpi)

    plt.figure(figsize=figsize)
    plt.title(title)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()

def to_gif(canvases, fn='animation.gif', duration=250):
    #imgs = [PIL.Image.fromarray((img.transpose((1,2,0))*255.).astype(np.uint8)) for img in canvases]
    imgs = []
    for i in range(len(canvases)):
        # np_img = (np.clip(canvases[i], 0, 1).transpose((1,2,0))*255.).astype(np.uint8)
        # print(canvases[i].max())
        if canvases[i].max() > 2:
            np_img = np.clip(canvases[i], 0, 255.)
        else:
            np_img = np.clip(canvases[i], 0, 1)*255.
        np_img = np_img.astype(np.uint8)

        imgs.append(PIL.Image.fromarray(np_img))
    # duration is the number of milliseconds between frames; this is 40 frames per second
    # imgs[0].save(fn, save_all=True, append_images=imgs[1:], duration=50, loop=0)
    imgs[0].save(fn, save_all=True, append_images=imgs[1:], duration=duration, loop=0)

def to_video(frames, fn='animation{}.mp4'.format(time.time()), frame_rate=10):
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


def save_colors(allowed_colors):
    """
    Save the colors used as an image so you know how to mix the paints
    args:
        allowed_colors (List((BGR),...) : List of BGR (Is it? it is) tuples
        actions (List(actions)) : list of 13 valued tuples. Used to determine how much each color is used
        output_dir (String) : Where to save the image
    """
    n_colors = len(allowed_colors)
    i = 0
    w = 128
    big_img = np.ones((2*w, 6*w, 3))

    for c in allowed_colors:
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

    color = [np.median(canvas_after[:,:,ch][stroke_bool_map]) for ch in range(3)]
    return np.array(color)

def load_instructions(fn):
    '''
    Load instructions into a list of lists
    '''

    instructions = []
    with open(fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 1:
                instructions.append(np.array([float(s) for s in line.split(',')]))
    return instructions

def get_colors(img, n_colors=6):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(img.reshape((img.shape[0]*img.shape[1],3)))
    colors = [kmeans.cluster_centers_[i] for i in range(len(kmeans.cluster_centers_))]
    return colors


def discretize_image_old(img, allowed_colors):
    """
    Only use allowed_colors in the given image. Use euclidean distance for speed.
    args:
        img (np.array[width, height, 3]) : target image 
        allowed_colors (List((R,G,B),...) : List of RGB tuples
    return:
        np.array[width, height, 3] : target image using only the allowed colors
    """
    n_pix = img.shape[0]*img.shape[1]
    n_colors = len(allowed_colors)

    img_flat = np.reshape(img, (n_pix, 3)) #/ 255.

    color_mat = np.empty((n_colors, n_pix, 3))

    i = 0
    for c in allowed_colors:
        color_mat[i] = np.tile(c[np.newaxis].T, (1, n_pix)).T
        i += 1

    img_exp = np.tile(img_flat[np.newaxis], (n_colors, 1, 1))
    img_exp = img_exp.astype(np.float32)
    color_mat = color_mat.astype(np.float32)
    diff = np.sum(np.abs(img_exp - color_mat), axis=2)

    argmin = np.argmin(diff, axis=0)

    img_disc = np.array(allowed_colors)[argmin]
    img_disc = np.reshape(img_disc, (img.shape[0],img.shape[1], 3))

    return img_disc

def discretize_image(img, allowed_colors):
    """
    Only use allowed_colors in the given image. Use euclidean distance for speed.
    args:
        img (np.array[width, height, 3]) : target image 
        allowed_colors (List((R,G,B),...) : List of RGB tuples
    return:
        np.array[width, height, 3] : target image using only the allowed colors
    """
    n_pix = img.shape[0]*img.shape[1]
    n_colors = len(allowed_colors)

    # print(np.reshape(img, (n_pix, 3)).shape)
    img_flat = np.reshape(rgb2lab(img), (n_pix, 3))

    diff = np.zeros((n_colors, n_pix), dtype=np.float32)

    i = 0
    for c in allowed_colors:
        c_mat = np.tile(c[np.newaxis].T, (1, n_pix)).T
        diff[i,:] = compare_images(rgb2lab(c_mat[np.newaxis])[0], img_flat)
        i += 1

    argmin = np.argmin(diff, axis=0)

    img_disc = np.array(allowed_colors)[argmin]
    img_disc = np.reshape(img_disc, (img.shape[0],img.shape[1], 3))

    return img_disc

def nearest_color(color, discrete_colors):
    ''' Get the most similar color to a given color (np.array([3])) '''
    dist = np.mean(np.abs(discrete_colors - color[None,:]), axis=1)
    argmin = np.argmin(dist)
    return argmin, discrete_colors[argmin]

def get_mixed_paint_colors(table_photo, n_colors, use_cache=False, cache_dir=None):
    plt.imshow(table_photo)
    plt.title("Click paints. Bottom left then up then right and up.")
    if use_cache and os.path.exists(os.path.join(cache_dir, 'palette_points.pkl')):
        points = pickle.load(open(os.path.join(cache_dir, "palette_points.pkl"),'rb'))
    else:
        points = np.array(plt.ginput(n=n_colors)).astype(np.int64)
        with open(os.path.join(cache_dir, 'palette_points.pkl'),'wb') as f:
            pickle.dump(points, f)
    t = 15
    real_colors = []
    for i in range(n_colors):
        y, x = points[i,1], points[i,0]
        real_colors.append(np.median(table_photo[y-t:y+t,x-t:x+t,:], axis=(0,1)))
    return np.array(real_colors)

def increase_brightness(img, value=30):
    # https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


from skimage.filters import laplace, sobel
def edge_loss(img0, img1):
    img0 = cv2.resize(img0, (256,256))
    img1 = cv2.resize(img1, (256,256))
    return np.abs(sobel(np.mean(img0, axis=2)) - sobel(np.mean(img1, axis=2)))

def load_img(fn, width=None, height=None):
    img = cv2.imread(fn)[:,:,::-1]
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height))
    return img
#! /usr/bin/env python
import os
import time
import sys
import cv2
import rospy
import numpy as np
from tqdm import tqdm
import scipy.special
from scipy.ndimage import median_filter
import argparse
import datetime

from paint_utils import *
from painter import Painter, canvas_to_global_coordinates, CANVAS_WIDTH, CANVAS_HEIGHT
from strokes import all_strokes
from simulated_painting_environment import apply_stroke
from robot import *
from strokes import *
from dslr import WebCam
from content_loss import * 
import torch
import lpips

from tensorboard import TensorBoard
date_and_time = datetime.datetime.now()
run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
writer = TensorBoard('painting/{}'.format(run_name))


import matplotlib
import matplotlib.pyplot as plt

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

def process_img_for_logging(img, max_size=256.):
    max_size *= 1.
    fact = img.shape[0] / max_size if img.shape[0] > max_size else 1.# Width to 512
    img = cv2.resize(img.copy(), (int(img.shape[1]/fact), int(img.shape[0]/fact)))
    return img

# def paint_in_simulation(target, canvas, painter, colors):
#     print("Painting in simulation")
#     canvas = canvas.astype(np.float32)
#     canvas /= 255.
#     for it in tqdm(range(2000)):
#         # color_ind = int(np.floor(it / 30.)%len(colors))
#         if it % 10==0: color_ind = color_that_can_help_the_most(
#             target, cv2.resize(canvas.copy()*255., (target.shape[1], target.shape[0])), colors)
#         color = colors[color_ind]
#         x, y, stroke_ind, rotation, canvas, loss, diff, stroke_bool_map \
#             = painter.next_stroke(canvas.copy()*255., target, color, x_y_attempts=1)

#         if it % 20 == 0:
#             writer.add_image('images/simulated', canvas, it)
#     return cv2.resize(canvas, (target.shape[1], target.shape[0]))


from skimage.filters import laplace, sobel
def edge_loss(img0, img1):
    img0 = cv2.resize(img0, (256,256))
    img1 = cv2.resize(img1, (256,256))
    return np.abs(sobel(np.mean(img0, axis=2)) - sobel(np.mean(img1, axis=2)))

def paint_in_simulation(target, canvas, painter, colors):
    weight = sobel(np.mean(cv2.resize(target, (256,256)), axis=2))
    weight = cv2.resize(weight, (target.shape[1], target.shape[0]))
    weight -= weight.min()
    weight /= weight.max()

    writer.add_image('target/edge_target', sobel(np.mean(cv2.resize(target, (256,256)), axis=2)), 0)
    print("Painting in simulation")
    canvas = canvas.astype(np.float32)
    canvas /= 255.
    for it in tqdm(range(2000)):
        # color_ind = int(np.floor(it / 30.)%len(colors))
        if it % 10==0: color_ind = color_that_can_help_the_most(
            target, cv2.resize(canvas.copy()*255., (target.shape[1], target.shape[0])), colors)
        color = colors[color_ind]
        x, y, stroke_ind, rotation, canvas, loss, diff, stroke_bool_map \
            = painter.next_stroke(canvas.copy()*255., target, color, x_y_attempts=1, 
                # weight=weight,
                # # loss_fcn=lambda c,t: np.sqrt(np.abs(c - t)),
                # loss_fcn = lambda c,t : np.mean(edge_loss(c, t))*5 + np.mean(np.abs(c - t)),
                )

        if it % 20 == 0:
            writer.add_image('images/simulated', canvas, it)
            writer.add_image('images/edge', sobel(np.mean(cv2.resize(canvas, (256,256)), axis=2))*255., it)
    return cv2.resize(canvas, (target.shape[1], target.shape[0]))


def get_mixed_paint_colors(table_photo, n_colors, use_cache=False):
    plt.imshow(table_photo)
    plt.title("Click paints. Bottom left then up then right and up.")
    if use_cache and os.path.exists('palette_points.pkl'):
        points = pickle.load(open("palette_points.pkl",'rb'))
    else:
        points = np.array(plt.ginput(n=n_colors)).astype(np.int64)
        with open('palette_points.pkl','wb') as f:
            pickle.dump(points, f)
    t = 15
    real_colors = []
    for i in range(n_colors):
        y, x = points[i,1], points[i,0]
        real_colors.append(np.median(table_photo[y-t:y+t,x-t:x+t,:], axis=(0,1)))
    return np.array(real_colors)

def discretize_image(img, allowed_colors):
    """
    Only use allowed_colors in the given image. Use euclidean distance for speed.
    args:
        img (np.array[width, height, 3]) : target image 
        allowed_colors (List((B,G,R),...) : List of RGB tuples
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

def color_that_can_help_the_most(target, canvas, colors):
    # largest_loss, best_color_ind = -1, None 
    # for color_ind in range(len(colors)):
    #     diff = np.mean(np.abs(canvas - target), axis=2) \
    #             * (255. - np.mean(np.abs(colors[color_ind][None,None,:] - target), axis=2))
    #     loss = np.mean(diff)
    #     if loss > largest_loss:
    #         largest_loss = loss 
    #         best_color_ind = color_ind 
    # return color_ind
    color_losses = np.zeros(len(colors), dtype=np.float32) # Distribution of losses
    for color_ind in range(len(colors)):
        diff = np.mean(np.abs(canvas - target), axis=2) \
                * (255. - np.mean(np.abs(colors[color_ind][None,None,:] - target), axis=2))
        color_losses[color_ind] = np.mean(diff)
    color_probabilities = color_losses / color_losses.sum() # To Distribution
    print(color_probabilities)
    color_ind = np.random.choice(len(colors), 1, p=color_probabilities)
    return int(color_ind[0])

def paint_coarsely(painter, target, colors):
    # Median filter target image so that it's not detailed
    og_shape = target.shape
    target = cv2.resize(target.copy(), (256,256)) # For scipy memory error
    target = median_filter(target, size=(int(target.shape[0]*0.1),int(target.shape[0]*0.1), 3))
    target = cv2.resize(target, (og_shape[1], og_shape[0]))
    # show_img(target/255., title="Target for the coarse painting phase")
    writer.add_image('target/coarse_target', target/255., 0)

    paint_planner(painter, target, colors,
            how_often_to_get_paint = 4,
            strokes_per_color = 12,
            camera_capture_interval = 12
    )


def paint_finely(painter, target, colors):
    # with torch.no_grad():
    #     target_square = cv2.resize(target.copy(), (256,256))
    #     target_tensor = torch.from_numpy(target_square.transpose(2,0,1)).unsqueeze(0).to(device).float()
    #     #print(target_tensor.shape)
    #     content_mask = get_l2_mask(target_tensor)
    #     #print(content_mask.shape)
    #     content_mask = content_mask.detach().cpu().numpy()[0,0]
    #     content_mask = cv2.resize(content_mask.copy(), (target.shape[1], target.shape[0]))
    #     writer.add_image('target/content_mask', content_mask*255., 0)

    paint_planner(painter, target, colors,
            how_often_to_get_paint = 4,
            strokes_per_color = 5,
            camera_capture_interval = 4,
            # loss_fcn = lambda c,t: np.abs(c - t) + lpips_loss(c, t)
            # loss_fcn = lambda c,t : np.mean(edge_loss(c, t))*5 + np.mean(np.abs(c - t))
        )

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

global_it = 0
def paint_planner(painter, target, colors, 
        how_often_to_get_paint,
        strokes_per_color,
        camera_capture_interval,
        weight=None, loss_fcn=None):
    camera_capture_interval = 1################

    curr_color = None
    canvas_after = painter.camera.get_canvas()
    consecutive_paints = 0 
    full_sim_canvas = canvas_after.copy()
    global global_it
    og_target = target.copy()
    for it in tqdm(range(2000)):
        canvas_before = canvas_after
        
        if it % strokes_per_color == 0:
            # color_ind = int(np.floor(it / 30. )%args.n_colors)
            # Pick which color to work with next based on how much it can help
            color_ind = 0
            while color_ind == 0:
                color_ind = color_that_can_help_the_most(target, canvas_before, colors)
            color = colors[color_ind]

        x, y, stroke_ind, rotation, sim_canvas, loss, diff, stroke_bool_map \
            = painter.next_stroke(canvas_before, target, color, 
                )

        full_sim_canvas, _ = apply_stroke(full_sim_canvas.copy(), painter.strokes[stroke_ind], stroke_ind, 
            colors[color_ind], int(x*target.shape[1]), int((1-y)*target.shape[0]), rotation)

        new_paint_color = color_ind != curr_color
        if new_paint_color:
            painter.clean_paint_brush()
            curr_color = color_ind

        if consecutive_paints >= how_often_to_get_paint or new_paint_color:
            painter.get_paint(color_ind)
            consecutive_paints = 0

        x,y,_ = canvas_to_global_coordinates(x,y,None)

        all_strokes[stroke_ind]().paint(painter, x, y, rotation * (2*3.14/360))
        if it % camera_capture_interval == 0:
            painter.to_neutral()
            canvas_after = painter.camera.get_canvas()

        # Update representation of paint color
        new_paint_color = extract_paint_color(canvas_before, canvas_after, stroke_bool_map)
        color_momentum = 0.1
        if new_paint_color is not None:
            colors[color_ind] = colors[color_ind] * (1-color_momentum) \
                              + new_paint_color * color_momentum

        consecutive_paints += 1

        
        writer.add_scalar('loss/sim_stroke', loss, global_it)
        writer.add_scalar('loss/actual_stroke', np.mean(np.abs(canvas_before - canvas_after)), global_it)
        writer.add_scalar('loss/loss', np.mean(np.abs(target - canvas_after)), global_it)
        writer.add_scalar('loss/sim_loss', np.mean(np.abs(target - full_sim_canvas)), global_it)
        writer.add_scalar('loss/sim_actual_diff', np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.)), global_it)

        if it % 2 == 0:
            writer.add_image('images/propsed_stroke', process_img_for_logging(sim_canvas), global_it)
            writer.add_image('images/actual_stroke', process_img_for_logging(canvas_after), global_it)
            writer.add_image('images/sim_canvas', process_img_for_logging(full_sim_canvas/255.), global_it)
            # writer.add_image('images/diff', process_img_for_logging(diff/255.), global_it)
            # writer.add_image('images/sim_actual_diff', 
            #     process_img_for_logging(np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.), axis=2)), global_it)

            all_colors = save_colors(colors)
            # show_img(all_colors/255., title="Mix these colors, then exit this popup to start painting")
            writer.add_image('paint_colors/are', all_colors/255., global_it)
        global_it += 1

        if it % 20 == 0:
            target = discretize_image(og_target, colors)
            writer.add_image('target/discrete', target/255., global_it) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sawyer Painter')

    parser.add_argument("--file", type=str,
        default='/home/peterschaldenbrand/Downloads/david_lynch.csv',
        help='Path CSV instructions.')

    parser.add_argument('--type', default='cubic_bezier', type=str, help='Type of instructions: [cubic_bezier | bezier]')
    parser.add_argument('--continue_ind', default=0, type=int, help='Instruction to start from. Default 0.')
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--n_colors', default=6, type=int, help='Number of colors of paint to use')

    args = parser.parse_args()
    # args.file = '/home/peterschaldenbrand/paint/AniPainter/animation_instructions/actions.csv'
    # args.file = '/home/peterschaldenbrand/Downloads/actions.csv'

    # Set which robot we talkin to
    # os.environ['ROS_MASTER_URI'] = "http://localhost:11311"
    # os.environ['ROS_HOSTNAME'] = "localhost"

    painter = Painter(robot="sawyer", use_cache=args.use_cache, camera=WebCam())

    painter.robot.display_frida()
    # target = cv2.imread('/home/frida/ros_ws/src/intera_sdk/SawyerPainter/scripts/cutoutBen.jpg')[:,:,::-1]
    # target = cv2.imread('/home/frida/Downloads/cutoutAbby.jpg')[:,:,::-1]
    # target = cv2.imread('/home/frida/Downloads/cutoutJean.jpg')[:,:,::-1]
    # target = cv2.imread('/home/frida/Downloads/frog.jpg')[:,:,::-1]
    # target = cv2.imread('/home/frida/Downloads/cutoutjon.jpg')[:,:,::-1]
    target = cv2.imread('/home/frida/Downloads/cutoutTanmay.jpg')[:,:,::-1]
    
    canvas = painter.camera.get_canvas()
    target = cv2.resize(target, (canvas.shape[1], canvas.shape[0]))
    target = np.array(target)
    writer.add_image('target/real', target, 0)
    full_sim_canvas = canvas.copy()

    colors = get_colors(cv2.resize(target, (512, 512)), n_colors=args.n_colors)
    colors = sorted(colors, key=lambda l:np.mean(l), reverse=True) # Light to dark
    all_colors = save_colors(colors)
    # show_img(all_colors/255., title="Mix these colors, then exit this popup to start painting")
    writer.add_image('paint_colors/should_be', all_colors/255., 0)


    # colors = get_mixed_paint_colors(painter.camera.get_color_correct_image(), args.n_colors, use_cache=args.use_cache)
    # all_colors = save_colors(colors)
    # show_img(all_colors/255.)
    # writer.add_image('paint_colors/actual', all_colors/255., 0)

    # Change target image to only use actual paint colors
    target_not_discrete = target.copy()
    target = discretize_image(target, colors)
    writer.add_image('target/discrete', target/255., 0)


    # Use simulated painting as target
    # target = paint_in_simulation(target_not_discrete, canvas, painter, colors)
    # writer.add_image('target/simulated', target, 0)
    # show_img(target/255., title="Simulated painting. Close this popup to start painting this.")

    # paint_coarsely(painter, target, colors)
    paint_finely(painter, target_not_discrete, colors)






    # painter.robot.take_picture()

    # instructions = load_instructions(args.file)

    # curr_color = -1
    # since_got_paint = 0
    
    # for instr in tqdm(instructions[args.continue_ind:]):
    #     if args.type == 'cubic_bezier':
    #         # Cubic Bezier
    #         path = instr[2:]
    #         path = np.reshape(path, (len(path)//2, 2))
    #         color = instr[1]
    #         radius = instr[0]
    #         if color != curr_color:
    #             painter.clean_paint_brush()

    #         if color != curr_color or since_got_paint == painter.GET_PAINT_FREQ:
    #             painter.get_paint(color)
    #             since_got_paint = 0
    #         since_got_paint += 1

    #         painter.paint_cubic_bezier(path)
    #         curr_color = color
    #     else:
    #         # Quadratic Bezier Curve
    #         p0, p1, p2 = instr[0:2], instr[2:4], instr[4:6]
    #         color = instr[12]
    #         radius = instr[6]
    #         if color != curr_color:
    #             painter.clean_paint_brush()

    #         if color != curr_color or since_got_paint == painter.GET_PAINT_FREQ:
    #             painter.get_paint(color)
    #             since_got_paint = 0
    #         since_got_paint += 1

    #         painter.paint_quadratic_bezier(p0, p1, p2)
    #         curr_color = color
    #     # take a picture
    # painter.clean_paint_brush()

    # # for i in range(12):
    # #     get_paint(i)
    # # get_paint(0)
    # # get_paint(5)
    # # get_paint(6)
    # # get_paint(11)

    # # clean_paint_brush()

    # # paint_bezier_curve((0,0),(0,0),(0,0))
    # # paint_bezier_curve((0,1),(0,1),(0,1)) # top-left
    # # paint_bezier_curve((1,0),(1,0),(1,0)) # bottom-right
    # # paint_bezier_curve((1,1),(1,1),(1,1))


    painter.to_neutral()


    painter.robot.good_night_robot()
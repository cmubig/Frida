
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################


import numpy as np
import torch
from torch import nn
import cv2
from tqdm import tqdm
import os
import time
import copy
# from paint_utils import save_colors

from options import Options

from painter import Painter

from paint_utils3 import to_video
from paint_utils3 import *
from torchvision.utils import save_image
from painter import canvas_to_global_coordinates

# from test_controlnet import pipeline as sd_interactive_pipeline

matplotlib.use('TkAgg')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

# Utilities

writer = None
local_it = 0 
plans = []

def log_progress(painting, log_freq=5, force_log=False, title='plan'):
    global local_it, plans
    local_it +=1
    if (local_it %log_freq==0) or force_log:
        with torch.no_grad():
            #np_painting = painting(h,w, use_alpha=False).detach().cpu().numpy()[0].transpose(1,2,0)
            #opt.writer.add_image('images/{}'.format(title), np.clip(np_painting, a_min=0, a_max=1), local_it)
            p = painting(h,w, use_alpha=False)
            p = format_img(p)
            opt.writer.add_image('images/{}'.format(title), p, local_it)
            
            plans.append((p*255.).astype(np.uint8))

def plan(target_image, robot, painter, full_res=256):
    # image is PIL Image
    image = target_image.resize((full_res, full_res))
    img_np = np.array(image)
    # edges = cv2.Canny(cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY),50, 200, 3)
    # print(edges.min(), edges.max())
    # edges[edges < 0.5*255] = 0
    # # plt.imshow(edges)
    # # plt.show()
    # print(edges.shape, edges.max(), edges.min())

    # Creating kernel 
    kernel = np.ones((3,3), np.uint8) 
    
    # Using cv2.erode() method  
    # img_np = cv2.erode(img_np, kernel)  
    img_np = cv2.dilate(img_np, kernel, iterations=1) 

    # plt.imshow(np.concatenate([np.array(image),img_np], axis=1))
    # plt.show()

    edges = img_np.astype(np.float32).mean(axis=2) / 255.
    edges = cv2.Canny(cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY),50, 200, 3)
    # diff = img_np.astype(np.float32).mean(axis=2) / 255.
    # edges = edges * diff
    # edges = 1 - edges
    edges[edges<0.2] = 0
    from scipy.ndimage import median_filter

    # # # edges = median_filter(edges, size=(11,11))
    # plt.imshow(edges)
    # plt.colorbar()
    # plt.show()

    points = (np.array(np.nonzero(edges))).astype(int)

    # Subsample
    sub_samp = list(np.arange(points.shape[1]))
    import random 
    random.shuffle(sub_samp)
    points = points[:,np.array(sub_samp[:10000])]
    print('points', points.shape, points.max(), points[:,0])

    # points = points[:,:2000]
    from python_tsp.exact import solve_tsp_dynamic_programming

    # distance_matrix = np.array([
    #     [0,  5, 4, 10],
    #     [5,  0, 8,  5],
    #     [4,  8, 0,  3],
    #     [10, 5, 3,  0]
    # ])
    from scipy.spatial import distance_matrix

    #for j in range(8):
    while(points.shape[1] > 1600):
    # for j in range(2):
        d_mat = distance_matrix(points.T, points.T)
        for i in range(len(d_mat)):
            d_mat[i,i] = 1e9

        min_dists = d_mat.argmin(axis=0)
        print('min_dists', min_dists.shape, min_dists[:50])
        min_inds = np.maximum(np.arange(points.shape[1]), min_dists)
        point_inds = np.unique(min_inds)
        points = points[:,point_inds]
        print('points', points.shape, point_inds.shape)
    # 1/0
    d_mat = distance_matrix(points.T, points.T)
    print('d_mat', d_mat.shape)
    
    from tsp_solver.greedy import solve_tsp
    path = solve_tsp(d_mat)
    # path = np.arange(points.shape[1])
    print('permutation', path[:10])

    img_np = np.array(image)
    img_np_gray = img_np.astype(float).mean(axis=2)
    img_np_gray /= img_np_gray.max()
    # img_np_gray[img_np_gray > 0.2] = 1
    img_np_gray = img_np_gray.astype(np.float32)
    # img_np_gray = median_filter(img_np_gray, size=(17,17))
    # plt.imshow(img_np_gray)
    # plt.colorbar()
    # plt.show()
    
    xs = [points[1,i] for i in path]
    ys = [points[0,i] for i in path]

    # plt.plot(xs, ys)
    # # plt.scatter(xs, ys)
    # plt.imshow(img_np)
    # plt.show()

    def move(real_x, real_y, real_z):
        robot.go_to_cartesian_pose([real_x, real_y, real_z],
                                   [0,0,0,0])

    def smooth(x, y, target):
        new_x, new_y = [], []
        for i in range(1,len(x)-1, 2):
            x_without = int((x[i-1] + x[i+1])/2)
            y_without = int((y[i-1] + y[i+1])/2)
            print(x_without, np.mean(target[y_without, x_without]))
            if (target[y_without, x_without])/256 < 0.2:
                new_x.append(x[i])
                new_y.append(y[i])
            else:
                new_x.append(x_without)
                new_y.append(y_without)
            new_x.append(x[i+1])
            new_y.append(y[i+1])
        return new_x, new_y
    

    xs, ys = smooth(xs, ys, np.array(image))

    plt.plot(xs, ys)
    # plt.scatter(xs, ys)
    plt.imshow(img_np)
    plt.show()


    strokes = []
    curr_stroke = None # np.array([n,2]) x,y
    n_points = 10
    prev_x, prev_y = -1, -1
    pen_up = False
    z_canvas = painter.Z_CANVAS 
    z_pen_up = z_canvas + 0.01

    for i in path:
        x = points[1,i]
        y = points[0,i]

        if prev_x > 0:
            pix_between = img_np_gray[
                   np.linspace(start=prev_y, stop=y, num=n_points).astype(int),
                   np.linspace(start=prev_x, stop=x, num=n_points).astype(int)
                ]
            avg_val = np.quantile(pix_between, 0.5)
            # print(avg_val)

            if avg_val > 0.95: # No Stroke
                if curr_stroke is not None:
                    strokes.append(curr_stroke)
                curr_stroke = None
            else: # Stroke continues
                if curr_stroke is None:
                    # start a new one
                    curr_stroke = np.array([[x,y]])
                else:
                    curr_stroke = np.concatenate([curr_stroke, np.array([[x,y]])], axis=0)
        else:
            # First point
            curr_stroke = np.array([[x,y]])
        prev_x, prev_y = x, y
    if curr_stroke is not None:
        strokes.append(curr_stroke)
    

    # Reduce stroke variation (they're too choppy)
    def smooth_stroke(stroke, target):
        for i in range(1, len(stroke)-1, 1):

    smoothed_strokes = []
    for stroke in strokes:

    # # print('n_strokes', len(strokes))
    # # plt.matshow(img_np_gray)
    # # for stroke in strokes:
    # #     plt.plot(stroke[:,0], stroke[:,1])
    # # # # plt.scatter(xs, ys)
    # # plt.show()
    
    # painting = Painting(opt)
    # renderer = painting.param2img.renderer

    # # trajectories: (n, 2, self.P)
    #     # thicknesses: (n, 1, self.P)
    # stroke = strokes[0]
    # s = renderer()
    # plt.matshow(s)
    # plt.show()

    
    xs = []
    ys = []
    n_points = 10
    prev_x, prev_y = -1, -1
    pen_up = False
    z_canvas = painter.Z_CANVAS 
    z_pen_up = z_canvas + 0.01

    for i in path:
        x = points[1,i]
        y = points[0,i]

        real_x, real_y = x.astype(np.float32)/full_res, y.astype(np.float32)/full_res
        real_x, real_y = min(max(real_x,0.),1.), min(max(real_y,0.),1.) #safety
        real_x, real_y,_ = canvas_to_global_coordinates(real_x,real_y,None,painter.opt)
        
        if prev_x > 0:
            #dist = ((x-prev_x)**2 + (y-prev_y**2))**0.5
            # avg_val = img_np_gray[
            #        np.linspace(start=prev_y, stop=y, num=n_points).astype(int),
            #        np.linspace(start=prev_x, stop=x, num=n_points).astype(int)
            #     ].mean()
            
            pix_between = img_np_gray[
                   np.linspace(start=prev_y, stop=y, num=n_points).astype(int),
                   np.linspace(start=prev_x, stop=x, num=n_points).astype(int)
                ]
            avg_val = np.quantile(pix_between, 0.15)
            # asdf = np.unique(np.linspace(start=prev_x, stop=x, num=n_points).astype(int))
            # if len(asdf) < 2:
            #     print('hey')
            # print(avg_val)

            if avg_val < 0.1:
                if pen_up:
                    move(real_x, real_y, z_pen_up)
                    #continue 
                else:
                    move(prev_real_x, prev_real_y, z_pen_up)
                    move(real_x, real_y, z_pen_up)
                    # print(xs)
                    # plt.plot(xs, ys)
                    # xs, ys = smooth(xs,ys, img_np_gray)
                    plt.plot(xs, ys)
                pen_up = True
                xs, ys = [x], [y]
            else:
                # print(avg_val)
                # Draw it
                if pen_up:
                    # MOve it down
                    move(prev_real_x, prev_real_y, z_canvas)
                pen_up = False
                xs.append(x)
                ys.append(y)
                move(real_x, real_y, z_canvas)
        else:
            xs.append(x)
            ys.append(y)
            move(real_x, real_y, z_pen_up)
            move(real_x, real_y, z_canvas)
        prev_x, prev_y = x, y
        prev_real_x, prev_real_y = real_x, real_y
    plt.plot(xs, ys)
    # plt.imshow(img_np_gray)
    # plt.imshow(img_np)
    plt.show()


if __name__ == '__main__':
    global opt
    opt = Options()
    opt.gather_options()


    # image = Image.open('/home/frida/Downloads/paint_this_img__another_copy_-removebg-preview.png')
    image = Image.open('/home/frida/Downloads/20230619_112748.jpg')

    opt.writer = create_tensorboard(log_dir=opt.tensorboard_dir)

    from robot import XArm, SimulatedRobot
    robot = SimulatedRobot() if opt.simulate else XArm(debug=True)
    robot.good_morning_robot()

    painter = Painter(opt)
    
    points = plan(image, robot, painter)
    robot.good_night_robot()

    for i in range(2):
        text_prompt = "Frog astronaut"#input("New text prompt:")

        curr_canvas = painter.camera.get_canvas()
        print(curr_canvas.shape, curr_canvas.max())
        curr_canvas_pil = Image.fromarray(curr_canvas.astype(np.uint8)).resize((512,512))
        with torch.no_grad():
            out_img = sd_interactive_pipeline(
                text_prompt, curr_canvas_pil, num_inference_steps=5, 
                # generator=generator,
                num_images_per_prompt=1,
                # controlnet_conditioning_scale=1.4,
            ).images[0]
        plt.imshow(out_img)
        plt.show()
    # to_video(plans, fn=os.path.join(opt.plan_gif_dir,'sim_canvases{}.mp4'.format(str(time.time()))))
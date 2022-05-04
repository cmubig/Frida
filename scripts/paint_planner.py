
##########################################################
#################### Copyright 2022 ######################
################ by Peter Schaldenbrand ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

import numpy as np
import cv2
import math
import copy
from tqdm import tqdm
from scipy.ndimage import median_filter

from simulated_painting_environment import apply_stroke
from painter import canvas_to_global_coordinates
from strokes import all_strokes
from paint_utils import *

def color_that_can_help_the_most(target, canvas, colors):
    target = cv2.resize(target.copy(), (512,512)) # FOr speed
    canvas = cv2.resize(canvas.copy(), (512,512))
    color_losses = np.zeros(len(colors), dtype=np.float32) # Distribution of losses
    for color_ind in range(len(colors)):
        diff = np.mean(np.abs(canvas - target), axis=2) \
                * (255. - np.mean(np.abs(colors[color_ind][None,None,:] - target), axis=2))
        color_losses[color_ind] = np.mean(diff)
    color_probabilities = color_losses / color_losses.sum() # To Distribution
    # print(color_probabilities)
    color_ind = np.random.choice(len(colors), 1, p=color_probabilities)
    return int(color_ind[0])

def paint_coarsely(painter, target, colors):
    # Median filter target image so that it's not detailed
    og_shape = target.shape
    target = cv2.resize(target.copy(), (256,256)) # For scipy memory error
    target = median_filter(target, size=(int(target.shape[0]*0.1),int(target.shape[0]*0.1), 3))
    target = cv2.resize(target, (og_shape[1], og_shape[0]))
    # show_img(target/255., title="Target for the coarse painting phase")
    painter.writer.add_image('target/coarse_target', target/255., 0)

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
    #     painter.writer.add_image('target/content_mask', content_mask*255., 0)

    paint_planner(painter, target, colors,
            how_often_to_get_paint = 4,
            strokes_per_color = 8,
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
    camera_capture_interval = 3################

    curr_color = None
    canvas_after = painter.camera.get_canvas()
    consecutive_paints = 0 
    full_sim_canvas = canvas_after.copy()
    global global_it
    og_target = target.copy()
    real_canvases = [canvas_after]
    sim_canvases = [canvas_after]

    for it in tqdm(range(2000)):
        canvas_before = canvas_after

        if it % strokes_per_color == 0:
            # color_ind = int(np.floor(it / 30. )%args.n_colors)
            # Pick which color to work with next based on how much it can help
            # color_ind = 0
            # while color_ind == 0:
            color_ind = color_that_can_help_the_most(target, canvas_before, colors)
            color = colors[color_ind]


        x, y, stroke_ind, rotation, sim_canvas, loss, diff, stroke_bool_map \
            = painter.next_stroke(canvas_before.copy(), target, color, x_y_attempts=2)

        full_sim_canvas, _, _ = apply_stroke(full_sim_canvas, painter.strokes[stroke_ind], stroke_ind,
            colors[color_ind], int(x*target.shape[1]), int((1-y)*target.shape[0]), rotation)

        new_paint_color = color_ind != curr_color
        if new_paint_color:
            painter.clean_paint_brush()
            curr_color = color_ind

        if consecutive_paints >= how_often_to_get_paint or new_paint_color:
            painter.get_paint(color_ind)
            consecutive_paints = 0

        x,y,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

        all_strokes[stroke_ind]().paint(painter, x, y, rotation * (2*3.14/360))

        if it % camera_capture_interval == 0:
            painter.to_neutral()
            canvas_after = painter.camera.get_canvas() if not painter.opt.simulate else full_sim_canvas
            real_canvases.append(canvas_after)
            sim_canvases.append(full_sim_canvas)

            # Update representation of paint color
            new_paint_color = extract_paint_color(canvas_before, canvas_after, stroke_bool_map)
            color_momentum = 0.1
            if new_paint_color is not None:
                colors[color_ind] = colors[color_ind] * (1-color_momentum) \
                                  + new_paint_color * color_momentum

        consecutive_paints += 1

        painter.writer.add_scalar('loss/loss', np.mean(np.abs(target - canvas_after)), global_it)
        painter.writer.add_scalar('loss/sim_loss', np.mean(np.abs(target - full_sim_canvas)), global_it)
        if not painter.opt.simulate:
            painter.writer.add_scalar('loss/sim_stroke', loss, global_it)
            painter.writer.add_scalar('loss/actual_stroke', np.mean(np.abs(canvas_before - canvas_after)), global_it)
            painter.writer.add_scalar('loss/sim_actual_diff', np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.)), global_it)

        if it % 5 == 0:
            if not painter.opt.simulate:
                painter.writer.add_image('images/canvas', canvas_after, global_it)
                painter.writer.add_image('images/propsed_stroke', sim_canvas, global_it)
                all_colors = save_colors(colors)
                # show_img(all_colors/255., title="Mix these colors, then exit this popup to start painting")
                painter.writer.add_image('paint_colors/are', all_colors/255., global_it)
            painter.writer.add_image('images/sim_canvas', full_sim_canvas/255., global_it)
            # painter.writer.add_image('images/diff', diff/255., global_it)
            # painter.writer.add_image('images/sim_actual_diff', 
            #     np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.), axis=2), global_it)

        global_it += 1

        # if it % 20 == 0:
        #     target = discretize_image(og_target, colors)
        #     painter.writer.add_image('target/discrete', target/255., global_it) 

        if it % 50 == 0:
            # to_gif(all_canvases)
            to_video(real_canvases, fn='real_canvases.mp4')
            to_video(sim_canvases, fn='sim_canvases.mp4')

def pick_next_stroke(curr_canvas, target, strokes, color, x_y_attempts, 
        H_coord=None, # Transform the x,y coord so the robot can actually hit the coordinate
        weight=None,
        loss_fcn=lambda c,t: np.mean(np.abs(c - t), axis=2)):
    """
    Given the current canvas and target image, pick the next brush stroke
    """
    # It's faster if the images are lower resolution
    fact = 8.#8.#8.
    curr_canvas = cv2.resize(curr_canvas.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))

    target = cv2.resize(target.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))
    strokes_resized = []
    for stroke in strokes:
        resized_stroke = cv2.resize(stroke.copy(), (int(stroke.shape[1]/fact), int(stroke.shape[0]/fact)))
        strokes_resized.append(resized_stroke)
    strokes = strokes_resized

    if weight is not None:
        weight = cv2.resize(weight.copy(), (target.shape[1], target.shape[0]))


    best_x, best_y, best_rot, best_stroke, best_canvas, best_loss \
        = None, None, None, None, None, 9999999
    best_stroke_ind = None
    best_stroke_bool_map = None

    # diff = np.mean(np.abs(curr_canvas - target), axis=2) \
    #         + (255. - np.mean(np.abs(color[None,None,:] - target), axis=2))
    diff = (255. - np.mean(np.abs(color[None,None,:] - target), axis=2)) # best in practice
    # diff = np.mean(np.abs(curr_canvas - target), axis=2) 

    if weight is not None:
        diff = diff * weight
    
    # Ignore edges
    diff[0:int(diff.shape[0]*0.05),:] = 0.
    diff[int(diff.shape[0] - diff.shape[0]*0.05):,:] = 0.
    diff[:,0:int(diff.shape[1]*0.05)] = 0.
    diff[:,int(diff.shape[1] - diff.shape[1]*0.05):] = 0.
    
    # Only look at indices where there is a big difference in canvas/target
    # good_y_inds, good_x_inds = np.where(diff > (np.quantile(diff, 0.9)-1e-3)) # 1e-3 to avoid where quantile is max value
    #diff = (diff > (np.quantile(diff, 0.9)-1e-3)).astype(np.float32)
    diff[diff < (np.quantile(diff,0.9)-1e-3)] = 0
    diff = diff/diff.sum() # turn to probability distribution

    # plt.imshow(diff)
    # plt.colorbar()
    # plt.show()
    target = target.astype(np.float32)
    # from skimage.filters import gaussian
    # blur_target = gaussian(target)
    color_help = np.mean(np.abs(color[None,None,:] - target), axis=2)
    color_help /= color_help.max()
    color_help = 1 - color_help
    
    loss_og = np.abs(target - curr_canvas)

    target_lab = rgb2lab(target)
    loss_og = compare_images(target_lab, rgb2lab(curr_canvas))

    for x_y_attempt in range(x_y_attempts): # Try a few random x/y's
        #x, y = np.random.randint(target.shape[1]), np.random.randint(target.shape[0])

        # ind = np.random.randint(len(good_x_inds))
        #x, y = good_x_inds[ind], good_y_inds[ind]  
        
        y, x = np.unravel_index(np.random.choice(len(diff.flatten()), p=diff.flatten()), diff.shape)

        if H_coord is not None:
            # Translate the coordinates so they're similar. see coordinate_calibration
            sim_coords = np.array([x * fact, y * fact, 1.])
            real_coords = H_coord.dot(sim_coords)
            x, y = real_coords[0]/real_coords[2]/fact, real_coords[1]/real_coords[2]/fact
            x, y = int(x), int(y)

        for stroke_ind in range(len(strokes)):
            stroke = strokes[stroke_ind]
            for rot in range(0, 360, 45):

                candidate_canvas, stroke_bool_map, bbox = \
                    apply_stroke(curr_canvas.copy(), stroke, stroke_ind, color, x, y, rot)

                # loss = np.mean(np.mean(np.abs(target - candidate_canvas), axis=2) * color_help)

                comp_inds = stroke_bool_map > 0.5
                # loss = np.mean(np.abs(target[comp_inds] - candidate_canvas[comp_inds]) * color_help[comp_inds][:,None] \
                #     - loss_og[comp_inds])
                loss = np.mean(compare_images(target_lab[comp_inds], rgb2lab(candidate_canvas)[comp_inds]) * color_help[comp_inds]\
                    - loss_og[comp_inds])

                if loss < best_loss:
                    best_loss = loss
                    best_x, best_y, best_rot, best_stroke, best_canvas \
                        = x, y, rot, stroke, candidate_canvas
                    best_stroke_ind = stroke_ind
                    best_stroke_bool_map = stroke_bool_map

                    # plt.imshow(loss_fcn(target, candidate_canvas.astype(np.float32)))
                    # plt.colorbar()
                    # plt.show()
                    # plt.imshow(np.mean(np.abs(target - candidate_canvas.astype(np.float32)), axis=2), cmap='gray')
                    # plt.colorbar()
                    # plt.show()

    return 1.*best_x/curr_canvas.shape[1], 1 - 1.*best_y/curr_canvas.shape[0],\
            best_stroke_ind, best_rot, best_canvas/255., \
            np.mean(np.abs(best_canvas - target)), \
            diff, best_stroke_bool_map

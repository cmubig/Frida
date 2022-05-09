
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


global_it = 0
def paint_planner(painter, target, colors, 
        how_often_to_get_paint,
        strokes_per_color,
        camera_capture_interval,
        loss_fcn=None):
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

        # Decide if and what color paint should be used next
        if it % strokes_per_color == 0:
            # color_ind = int(np.floor(it / 30. )%args.n_colors)
            # Pick which color to work with next based on how much it can help
            # color_ind = 0
            # while color_ind == 0:
            color_ind = color_that_can_help_the_most(target, canvas_before, colors)
            color = colors[color_ind]

        # Plan the next brush stroke
        x, y, stroke_ind, rotation, sim_canvas, loss, diff, stroke_bool_map \
            = painter.next_stroke(canvas_before.copy(), target, color, x_y_attempts=5)

        # Make the brush stroke on a simulated canvas
        full_sim_canvas, _, _ = apply_stroke(full_sim_canvas, painter.strokes[stroke_ind], stroke_ind,
            colors[color_ind], int(x*target.shape[1]), int((1-y)*target.shape[0]), rotation)

        # Clean paint brush and/or get more paint
        new_paint_color = color_ind != curr_color
        if new_paint_color:
            painter.clean_paint_brush()
            curr_color = color_ind
        if consecutive_paints >= how_often_to_get_paint or new_paint_color:
            painter.get_paint(color_ind)
            consecutive_paints = 0

        # Convert the canvas proportion coordinates to meters from robot
        x,y,_ = canvas_to_global_coordinates(x,y,None,painter.opt)

        # Paint the brush stroke
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

            diff = diff.astype(np.float32) / diff.max() *255. # Normalize for the image
            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
            diff[:30,:50,:] = color[None, None, :] # Put the paint color in the corner
            painter.writer.add_image('images/diff', diff/255., global_it)
            # painter.writer.add_image('images/sim_actual_diff', 
            #     np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.), axis=2), global_it)
        if it == 0:
            target_edges = sobel(np.mean(cv2.resize(target, (int(target.shape[1]/8), int(target.shape[0]/8))), axis=2))
            target_edges -= target_edges.min()
            target_edges /= target_edges.max()
            painter.writer.add_image('target/target_edges', target_edges*255., global_it)
        global_it += 1
        consecutive_paints += 1

        # if it % 20 == 0:
        #     target = discretize_image(og_target, colors)
        #     painter.writer.add_image('target/discrete', target/255., global_it) 

        if it % 50 == 0:
            # to_gif(all_canvases)
            to_video(real_canvases, fn='real_canvases.mp4')
            to_video(sim_canvases, fn='sim_canvases.mp4')


# from scipy.signal import medfilt

def pick_next_stroke(curr_canvas, target, strokes, color, x_y_attempts,
        H_coord=None, # Transform the x,y coord so the robot can actually hit the coordinate
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

    opt_params = { # Find the optimal parameters
        'x':None,
        'y':None,
        'rot':None,
        'stroke':None,
        'canvas':None,
        'loss':9999999,
        'stroke_ind':None,
        'stroke_bool_map':None,
    }

    target_lab = rgb2lab(target)

    target_edges = sobel(np.mean(target, axis=2))
    target_edges -= target_edges.min()
    target_edges /= target_edges.sum()

    # Regions of the canvas that the paint color matches well are high
    color_diff = compare_images(target_lab, rgb2lab(np.ones(target.shape) * color[None,None,:]))
    color_diff = color_diff.max() - color_diff # Small diff means the paint works well there

    loss_og = compare_images(target_lab, rgb2lab(curr_canvas))

    # Diff is a combination of canvas areas that are highly different from target
    # and where the paint color could really help the canvas
    diff = (loss_og / loss_og.sum()) * (color_diff / color_diff.sum())

    # Ignore edges for now
    diff[0:int(diff.shape[0]*0.05),:] = 0.
    diff[int(diff.shape[0] - diff.shape[0]*0.05):,:] = 0.
    diff[:,0:int(diff.shape[1]*0.05)] = 0.
    diff[:,int(diff.shape[1] - diff.shape[1]*0.05):] = 0.

    # Only look at indices where there is a big difference in canvas/target
    diff[diff < (np.quantile(diff,0.9))] = 0
    # diff = medfilt(diff,kernel_size=5)
    # # turn to probability distribution
    diff = diff/diff.sum()

    color_help = color_diff

    for x_y_attempt in range(x_y_attempts): # Try a few random x/y's
        # Randomly choose x,y locations to start the stroke weighted by the difference in canvas/target wrt color
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

                comp_inds = stroke_bool_map > 0.2 # only need to look where the stroke was made
                # loss = np.mean(np.abs(target[comp_inds] - candidate_canvas[comp_inds]) * color_help[comp_inds][:,None] \
                #     - loss_og[comp_inds])
                # loss = np.mean(compare_images(target_lab[comp_inds], rgb2lab(candidate_canvas)[comp_inds]) * color_help[comp_inds]\
                #     - loss_og[comp_inds])

                # Loss is the amount of difference lost
                loss = -1. * np.sum(diff[comp_inds])

                # Penalize crossing edges
                # loss += np.mean(target_edges[comp_inds])

                if loss < opt_params['loss']:
                    opt_params['x'], opt_params['y'] = x, y
                    opt_params['rot'] = rot
                    opt_params['stroke'] = stroke
                    opt_params['canvas'] = candidate_canvas
                    opt_params['loss'] = loss
                    opt_params['stroke_ind'] = stroke_ind
                    opt_params['stroke_bool_map'] = stroke_bool_map

    return 1.*opt_params['x']/curr_canvas.shape[1], 1 - 1.*opt_params['y']/curr_canvas.shape[0],\
            opt_params['stroke_ind'], opt_params['rot'], opt_params['canvas']/255., \
            np.mean(np.abs(opt_params['canvas'] - target)), \
            diff, opt_params['stroke_bool_map']

# from multiprocessing.pool import ThreadPool


#     pool = ThreadPool(20)
#     results = []
#                 results.append(pool.apply_async(apply_stroke,
#                     args=(curr_canvas.copy(), stroke, stroke_ind, color, x, y, rot)))
#                 params.append((stroke_ind, color, x, y, rot, stroke))


#     pool.close()
#     pool.join()
#     #print(results[0])
#     results = [r.get() for r in results]

#     #for r in results:
#     for i in range(len(results)):
#         candidate_canvas, stroke_bool_map, bbox = results[i]
#         p = params[i]
#         comp_inds = stroke_bool_map > 0.5
#         # loss = np.mean(np.abs(target[comp_inds] - candidate_canvas[comp_inds]) * color_help[comp_inds][:,None] \
#         #     - loss_og[comp_inds])
#         loss = np.mean(compare_images(target_lab[comp_inds], rgb2lab(candidate_canvas)[comp_inds]) * color_help[comp_inds]\
#             - loss_og[comp_inds])

#         if loss < best_loss:
#             best_loss = loss
#             best_x, best_y, best_rot, best_stroke, best_canvas \
#                 = p[2], p[3], p[4], p[5], candidate_canvas
#             best_stroke_ind = p[0]
#             best_stroke_bool_map = stroke_bool_map

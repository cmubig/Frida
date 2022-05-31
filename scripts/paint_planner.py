
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
import sys
import subprocess
from tqdm import tqdm
from scipy.ndimage import median_filter
from PIL import Image

from simulated_painting_environment import apply_stroke
from painter import canvas_to_global_coordinates
from strokes import all_strokes
from paint_utils import *

def parse_csv_line(line, painter):
    toks = line.split(',')
    if len(toks) != 7:
        return None
    x = int(float(toks[0])*painter.opt.CANVAS_WIDTH_PIX)
    y = int(float(toks[1])*painter.opt.CANVAS_HEIGHT_PIX)
    r = float(toks[2])*(360/(2*3.14))
    stroke_ind = int(toks[3])
    color = np.array([float(toks[4]), float(toks[5]), float(toks[6])])*255.
    return x, y, r, stroke_ind, color

global_it = 0
def paint_planner_new(painter, target, colors, how_often_to_get_paint=4):
    global global_it
    canvas_after = painter.camera.get_canvas()
    full_sim_canvas = canvas_after.copy()
    real_canvases = [canvas_after]
    sim_canvases = [canvas_after]
    consecutive_paints = 0
    target_lab = rgb2lab(target)
    curr_color = -1

    for it in tqdm(range(20)):
        canvas_before = canvas_after#painter.camera.get_canvas()

        # Save strokes for planning python file
        im = Image.fromarray(canvas_before.astype(np.uint8))
        im.save(os.path.join(painter.opt.cache_dir, 'current_canvas.jpg'))

        # Plan the new strokes
        exit_code = subprocess.call(['python3', 'plan_all_strokes.py']+sys.argv[1:])
        #print('exit code', exit_code)

        # Run Planned Strokes
        with open(os.path.join(painter.opt.cache_dir, "next_brush_strokes.csv"), 'r') as fp:
            for line in fp.readlines():
                canvas_before = canvas_after
                x, y, r, stroke_ind, color = parse_csv_line(line, painter)

                color_ind, color_discrete = nearest_color(color, colors)

                # Make the brush stroke on a simulated canvas
                full_sim_canvas, _, _ = apply_stroke(full_sim_canvas, painter.strokes[stroke_ind], stroke_ind,
                    colors[color_ind], x, y, r)

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
                all_strokes[stroke_ind]().paint(painter, x, y, r * (2*3.14/360))


                canvas_after = painter.camera.get_canvas() if not painter.opt.simulate else full_sim_canvas
                if global_it%1==0: # loggin be sloggin
                    painter.writer.add_scalar('loss/loss',
                        np.mean(compare_images(cv2.resize(target_lab, (256,256)), rgb2lab(cv2.resize(canvas_after, (256,256))))), global_it)
                    if not painter.opt.simulate:
                        painter.writer.add_scalar('loss/sim_loss',
                            np.mean(compare_images(cv2.resize(target_lab, (256,256)), rgb2lab(cv2.resize(full_sim_canvas, (256,256))))), global_it)

                        painter.writer.add_image('images/canvas', canvas_after/255., global_it)
                        all_colors = save_colors(colors)
                        painter.writer.add_image('paint_colors/are', all_colors/255., global_it)
                    painter.writer.add_image('images/sim_canvas', full_sim_canvas/255., global_it)

                global_it += 1
                consecutive_paints += 1


        if it % 100 == 0:
            # to_gif(all_canvases)
            to_video(real_canvases, fn='real_canvases.mp4')
            to_video(sim_canvases, fn='sim_canvases.mp4')


def color_that_can_help_the_most(target, canvas, colors):
    target = cv2.resize(target.copy(), (256,256)) # FOr speed
    canvas = cv2.resize(canvas.copy(), (256,256))

    target = discretize_image(target, colors)
    canvas = discretize_image(canvas, colors)
    # show_img(target/255.)
    # show_img(canvas/255.)

    color_probabilities = []
    for color_ind in range(len(colors)):
        color_pix = target == colors[color_ind][None,None,:]
        color_pix = np.all(color_pix, axis=2)
        #show_img(color_pix)
        #print(color_pix.astype(np.float32).sum(), color_pix.shape)
        # Find how many pixels in the canvas are incorrectly colored
        incorrect_colored_pix = (target[color_pix] != canvas[color_pix]).astype(np.float32).sum()
        color_probabilities.append(incorrect_colored_pix)
    color_probabilities = np.array(color_probabilities)
    #print(color_probabilities)
    color_probabilities /= color_probabilities.sum()

    #print(color_probabilities)
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
            camera_capture_interval = 4
        )


global_it = 0
def paint_planner(painter, target, colors,
        how_often_to_get_paint,
        strokes_per_color,
        camera_capture_interval,
        loss_fcn=None):
    camera_capture_interval = 3################

    curr_color = None
    canvas_after = painter.camera.get_canvas()*0. + 255.
    consecutive_paints = 0 
    full_sim_canvas = canvas_after.copy()
    global global_it
    og_target = target.copy()
    real_canvases = [canvas_after]
    sim_canvases = [canvas_after]
    target_lab = rgb2lab(target)


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
        x, y, stroke_ind, rotation, sim_canvas, loss, diff, stroke_bool_map, int_loss \
            = painter.next_stroke(canvas_before.copy(), target, color, colors, x_y_attempts=6)
        if int_loss > 0:
            # print(loss, 'diddnt help')
            continue # Stroke doesn't actually help

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
            #painter.to_neutral()
            canvas_after = full_sim_canvas#painter.camera.get_canvas() if not painter.opt.simulate else full_sim_canvas
            real_canvases.append(canvas_after)
            sim_canvases.append(full_sim_canvas)

            # Update representation of paint color
            new_paint_color = extract_paint_color(canvas_before, canvas_after, stroke_bool_map)
            color_momentum = 0.15
            if new_paint_color is not None:
                colors[color_ind] = colors[color_ind] * (1-color_momentum) \
                                  + new_paint_color * color_momentum


        if it%5==0: # loggin be sloggin
            painter.writer.add_scalar('loss/loss', 
                np.mean(compare_images(cv2.resize(target_lab, (256,256)), rgb2lab(cv2.resize(canvas_after, (256,256))))), global_it)
            if not painter.opt.simulate:
                painter.writer.add_scalar('loss/sim_loss', 
                    np.mean(compare_images(cv2.resize(target_lab, (256,256)), rgb2lab(cv2.resize(full_sim_canvas, (256,256))))), global_it)
                # painter.writer.add_scalar('loss/sim_stroke', loss, global_it)
                # painter.writer.add_scalar('loss/actual_stroke', np.mean(np.abs(canvas_before - canvas_after)), global_it)
                # painter.writer.add_scalar('loss/sim_actual_diff', \
                #     np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.)), global_it)

        if it % 5 == 0:
            if not painter.opt.simulate:
                painter.writer.add_image('images/canvas', canvas_after/255., global_it)
                # painter.writer.add_image('images/propsed_stroke', sim_canvas, global_it)
                all_colors = save_colors(colors)
                painter.writer.add_image('paint_colors/are', all_colors/255., global_it)
            painter.writer.add_image('images/sim_canvas', full_sim_canvas/255., global_it)

            diff = diff.astype(np.float32) / diff.max() *255. # Normalize for the image
            diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
            diff[:30,:50,:] = color[None, None, :] # Put the paint color in the corner
            painter.writer.add_image('images/diff', diff/255., global_it)
            # painter.writer.add_image('images/sim_actual_diff', 
            #     np.mean(np.abs(cv2.resize(canvas_after, (sim_canvas.shape[1], sim_canvas.shape[0])) - sim_canvas*255.), axis=2), global_it)
        
            canvas_edges = sobel(np.mean(canvas_after, axis=2))
            canvas_edges -= canvas_edges.min()
            canvas_edges /= canvas_edges.max()
            painter.writer.add_image('images/canvas_edges', canvas_edges*255., global_it)
        if it == 0:
            target_edges = sobel(np.mean(cv2.resize(target, (canvas_after.shape[1], canvas_after.shape[0])), axis=2))
            target_edges -= target_edges.min()
            target_edges /= target_edges.max()
            painter.writer.add_image('target/target_edges', target_edges*255., global_it)
        global_it += 1
        consecutive_paints += 1

        # if it % 20 == 0:
        #     target = discretize_image(og_target, colors)
        #     painter.writer.add_image('target/discrete', target/255., global_it) 

        if it % 100 == 0:
            # to_gif(all_canvases)
            to_video(real_canvases, fn='real_canvases.mp4')
            to_video(sim_canvases, fn='sim_canvases.mp4')


from scipy.signal import medfilt

# from content_loss import get_l2_mask
# import torch
# target_weight = None

target_edges = None

def pick_next_stroke(curr_canvas, target, strokes, color, colors, x_y_attempts, 
        H_coord=None, # Transform the x,y coord so the robot can actually hit the coordinate
        loss_fcn=lambda c,t: np.mean(np.abs(c - t), axis=2)):
    """
    Given the current canvas and target image, pick the next brush stroke
    """
    # It's faster if the images are lower resolution
    fact = target.shape[1] / 196.0 #12.#8.#8.
    curr_canvas = cv2.resize(curr_canvas.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))

    target = cv2.resize(target.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))
    strokes_resized = []
    for stroke in strokes:
        resized_stroke = cv2.resize(stroke.copy(), (int(stroke.shape[1]/fact), int(stroke.shape[0]/fact)))
        strokes_resized.append(resized_stroke)
    strokes = strokes_resized

    # global target_weight 
    # if target_weight is None or target_weight.shape[0] != target.shape[0]:
    #     target_weight = get_l2_mask(target) #+ 0.25

    global target_edges
    if target_edges is None or target_edges.shape[1] != target.shape[1]:
        target_edges = sobel(np.mean(target, axis=2))
        target_edges -= target_edges.min()
        target_edges /= target_edges.max()

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
    curr_canvas_lab = rgb2lab(curr_canvas)


    # Regions of the canvas that the paint color matches well are high
    # color_diff = compare_images(target_lab, rgb2lab(np.ones(target.shape) * color[None,None,:]))
    # color_diff = color_diff.max() - color_diff # Small diff means the paint works well there


    target_discrete = discretize_image(target, colors)
    canvas_discrete = discretize_image(curr_canvas, colors)

    color_pix = np.isclose(target_discrete, np.ones(target.shape) * color[None,None,:])
    color_pix = np.all(color_pix, axis=2).astype(np.float32)
    # Find how many pixels in the canvas are incorrectly colored
    color_diff = np.all(target_discrete != canvas_discrete, axis=2).astype(np.float32)*color_pix \
            + 1e-10
    #show_img(diff)

    loss_og = compare_images(target_lab, rgb2lab(curr_canvas)) 

    # Diff is a combination of canvas areas that are highly different from target
    # and where the paint color could really help the canvas
    diff = (loss_og / loss_og.sum()) * color_diff#(color_diff / color_diff.sum())

    # Ignore edges for now
    t = 0.01
    diff[0:int(diff.shape[0]*t),:] = 0.
    diff[int(diff.shape[0] - diff.shape[0]*t):,:] = 0.
    diff[:,0:int(diff.shape[1]*t)] = 0.
    diff[:,int(diff.shape[1] - diff.shape[1]*t):] = 0.

    # Median filter for smooothness
    # diff = medfilt(diff,kernel_size=5)




    # Turn to probability distribution
    if diff.sum() > 0:
        diff = diff/diff.sum()
    else:
        diff = np.ones(diff.shape, dtype=np.float32) / (diff.shape[0]*diff.shape[1])

    # diff = diff * target_weight
    # diff = diff/diff.sum()

    loss_before = np.sum(compare_images(curr_canvas_lab, target_lab))
    for x_y_attempt in range(x_y_attempts): # Try a few random x/y's
        # Randomly choose x,y locations to start the stroke weighted by the difference in canvas/target wrt color
        y, x = np.unravel_index(np.random.choice(len(diff.flatten()), p=diff.flatten()), diff.shape)
        # y, x = np.random.randint(diff.shape[0]), np.random.randint(diff.shape[1])

        for stroke_ind in range(len(strokes)):
            stroke = strokes[stroke_ind]
            for rot in range(0, 360, 30):

                candidate_canvas, stroke_bool_map, bbox = \
                    apply_stroke(curr_canvas.copy(), stroke, stroke_ind, color, x, y, rot)

                comp_inds = stroke_bool_map > 0.1 # only need to look where the stroke was made
                if comp_inds.sum() == 0:
                    # Likely trying to just paint near the edge of the canvas and rotation took the stroke
                    # off the canvas
                    print('No brush stroke region. Resolution probably too small.')
                    continue

                # Loss is the amount of difference lost
                # loss = -1. * np.sum(diff[comp_inds])

                # Loss is the L1 loss lost
                # loss = np.sum(np.abs(candidate_canvas[comp_inds] - target[comp_inds])) \
                #         - np.sum(np.abs(curr_canvas[comp_inds] - target[comp_inds]))
                #print(target_lab[comp_inds].shape, rgb2lab(candidate_canvas[comp_inds][np.newaxis])[0].shape)
                loss = np.sum(compare_images(rgb2lab(candidate_canvas[comp_inds][np.newaxis])[0], target_lab[comp_inds])) \
                        - np.sum(compare_images(curr_canvas_lab[comp_inds], target_lab[comp_inds]))
                # loss = np.sum(compare_images(rgb2lab(candidate_canvas), target_lab)) \
                #         - loss_before#np.sum(compare_images(curr_canvas_lab, target_lab))
                # loss = np.sum(diff[comp_inds]*(compare_images(rgb2lab(candidate_canvas[comp_inds][np.newaxis])[0], target_lab[comp_inds]) \
                #         - compare_images(curr_canvas_lab[comp_inds], target_lab[comp_inds])))
                
                # loss = np.mean(np.mean(np.abs(candidate_canvas - target), axis=2)* (1. - np.all(target == color, axis=2).astype(np.float32)))

                # Should have the same edges in target and canvas
                # canvas_edges = sobel(np.mean(candidate_canvas, axis=2))
                # canvas_edges -= canvas_edges.min()
                # canvas_edges /= canvas_edges.max()
                # edge_loss_weight = 100.0#1.0e-3
                # # edge_loss = np.mean(np.abs(canvas_edges - target_edges)*target_edges) * edge_loss_weight
                # edge_loss = np.mean(np.abs(canvas_edges - target_edges)) * edge_loss_weight
                # loss += edge_loss

                # Penalize making mistakes (painting in areas it shouldn't)
                # mistakes = compare_images(target_lab[comp_inds], \
                #     rgb2lab(candidate_canvas[comp_inds][np.newaxis])[0])
                # mistake_weight = .5#global_it/50.#1.2
                # loss += (mistakes.sum() / comp_inds.sum()) / 255. * mistake_weight


                if loss < opt_params['loss']:
                    opt_params['x'], opt_params['y'] = x, y
                    opt_params['rot'] = rot
                    opt_params['stroke'] = stroke
                    opt_params['canvas'] = candidate_canvas
                    opt_params['loss'] = loss
                    opt_params['stroke_ind'] = stroke_ind
                    opt_params['stroke_bool_map'] = stroke_bool_map
    if opt_params['loss'] > 0:
        print(opt_params['loss'])
    return 1.*opt_params['x']/curr_canvas.shape[1], 1 - 1.*opt_params['y']/curr_canvas.shape[0],\
            opt_params['stroke_ind'], opt_params['rot'], opt_params['canvas']/255., \
            np.mean(compare_images(target_lab, rgb2lab(opt_params['canvas']))), \
            diff, opt_params['stroke_bool_map'], opt_params['loss']


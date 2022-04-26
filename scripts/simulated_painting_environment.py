#! /usr/bin/env python3

import numpy as np
import cv2
import math
import copy
from scipy import ndimage

# Save some processed strokes to save time
# {stroke_ind+'_'+rotation+'_'+stroke_size: stroke nd.array}}
processed_stroke_cache = {}

def apply_stroke(canvas, stroke, stroke_ind, color, x, y, theta=0):
    '''
    args:
        canvas (np.array[width, height, 3]) : Current painting canvas 0-1 RGB
        stroke (np.array[width, height]) :  Stroke 0-1 intesity map
        color (np.array[3]) : RGB color to use for the brush stroke
        x, y (int) : location to apply the stroke in pixels. The start of the brush stroke.
        theta (float) : degrees of rotation
    '''
    # how to get to the start of the brush stroke from the top left of the cut out region
    down = 0.5
    right = 0.2
    
    stroke_cache_key = str(stroke_ind)+'_'+str(theta)+'_'+str(stroke.shape[0])

    if stroke_cache_key in processed_stroke_cache:
        stroke = processed_stroke_cache[stroke_cache_key].copy()
    else:
        # Padding for rotation. Ensure start of brush stroke is centered in square image
        h, w = stroke.shape
        padX = int((1 - right - right)*w), 0
        padY = int((1 - right)*w - (1 - down)*h), int((1 - right)*w - (1 - down)*h)
        imgP = np.pad(stroke, [padY, padX], 'constant')
        # show_img(imgP)

        # Rotate theta degrees
        stroke = ndimage.rotate(imgP, theta, reshape=False)
        # show_img(stroke)

        processed_stroke_cache[stroke_cache_key] = stroke.copy()

    
    s_expanded = np.tile(stroke[:,:, np.newaxis], (1,1,3))
    s_color = s_expanded * color[None, None, :]
    stroke_dim = stroke.shape
    
    h, w = stroke.shape # new height and width (with padding. should be equal)
    
    # Get indices of the canvas to apply the stroke to
    y_s, y_e = max(y - int(.5 * h), 0), min(y - int(.5 * h) + stroke_dim[0], canvas.shape[0])
    x_s, x_e = max(x - int(.5 * w), 0), min(x - int(.5 * w) + stroke_dim[1], canvas.shape[1])
    
    # Get the indices of the stroke to apply to the canvas (incase stroke falls off side of canvas)
    sy_s = max(0, -1 * (y - int(.5 * h)))
    sy_e = y_e - y_s - min(0, (y - int(.5 * h)))
    sx_s = max(0, -1 * (x - int(.5 * w)))
    sx_e = x_e - x_s - min(0, (x - int(.5 * w)))
#     print(y_s, y_e, sy_s, sy_e)
#     print(x_s, x_e, sx_s, sx_e)
    # Apply the stroke to the canvas
    canvas[y_s:y_e,x_s:x_e,:] \
        = canvas[y_s:y_e,x_s:x_e,:] * (1 - s_expanded[sy_s:sy_e,sx_s:sx_e]) + s_color[sy_s:sy_e,sx_s:sx_e]

    # plt.imshow(s_expanded)
    # plt.show()
    stroke_bool_map = np.zeros((canvas.shape[0], canvas.shape[1]))
    stroke_bool_map[y_s:y_e,x_s:x_e] = s_expanded[sy_s:sy_e,sx_s:sx_e,0]

    # plt.imshow(stroke_bool_map)
    # plt.colorbar()
    # plt.show()
    return canvas, stroke_bool_map


# def pick_next_stroke(curr_canvas, target, strokes, colors):
#     """
#     Given the current canvas and target image, pick the next brush stroke
#     """
#     # It's faster if the images are lower resolution
#     fact = 12
#     curr_canvas = cv2.resize(curr_canvas.copy(), (int(curr_canvas.shape[1]/fact), int(curr_canvas.shape[0]/fact)))
#     target = cv2.resize(target.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))
#     strokes_resized = []
#     for stroke in strokes:
#         resized_stroke = cv2.resize(stroke.copy(), (int(stroke.shape[1]/fact), int(stroke.shape[0]/fact)))
#         strokes_resized.append(resized_stroke)
#     strokes = strokes_resized

#     best_x, best_y, best_rot, best_stroke, best_canvas, best_loss \
#         = None, None, None, None, None, 9999999
#     best_color, best_color_ind = None, None
#     best_stroke_ind = None
#     for x_y_attempt in range(7): # Try a few random x/y's
#         x, y = np.random.randint(target.shape[1]), np.random.randint(target.shape[0])
#         #color = target[y, x]
        
#         for color_ind in range(len(colors)):
#             color = colors[color_ind]
#             #for stroke in strokes:
#             for stroke_ind in range(len(strokes)):
#                 stroke = strokes[stroke_ind]
#                 for rot in range(0, 360, 45):
#                     #print(curr_canvas.max(), stroke.max(), x, y, color)
#                     candidate_canvas = apply_stroke(curr_canvas.copy(), stroke, 
#                         color, x, y, rot)

#                     loss = np.mean(np.abs(target - candidate_canvas))
#                     #print(loss)
#                     if loss < best_loss:
#                         best_loss = loss
#                         best_x, best_y, best_rot, best_stroke, best_canvas \
#                             = x, y, rot, stroke, candidate_canvas
#                         best_color, best_color_ind = color, color_ind
#                         best_stroke_ind = stroke_ind
    
#     # show_img(best_canvas/255.)
#     return 1.*best_x/curr_canvas.shape[1], 1 - 1.*best_y/curr_canvas.shape[0],\
#             best_stroke_ind, best_color_ind, best_rot, best_canvas/255., best_loss

resized_target = None # Cache
resized_strokes = None # Cache
resized_weight = None # Cache
def pick_next_stroke(curr_canvas, target, strokes, color, x_y_attempts, 
        weight=None,
        loss_fcn=lambda c,t: np.mean(np.abs(c - t), axis=2)):
    """
    Given the current canvas and target image, pick the next brush stroke
    """
    # It's faster if the images are lower resolution
    fact = 8
    curr_canvas = cv2.resize(curr_canvas.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))
    global resized_target, resized_strokes, resized_weight
    if resized_target is None:
        target = cv2.resize(target.copy(), (int(target.shape[1]/fact), int(target.shape[0]/fact)))
        strokes_resized = []
        for stroke in strokes:
            resized_stroke = cv2.resize(stroke.copy(), (int(stroke.shape[1]/fact), int(stroke.shape[0]/fact)))
            strokes_resized.append(resized_stroke)
        strokes = strokes_resized

        if weight is not None:
            weight = cv2.resize(weight.copy(), (target.shape[1], target.shape[0]))

        resized_target = target  # Cache
        resized_strokes = strokes # Cache
        resized_weight = weight # Cache
    else:
        strokes = resized_strokes # Cache
        target = resized_target # Cache
        weight = resized_weight # Cache

    best_x, best_y, best_rot, best_stroke, best_canvas, best_loss \
        = None, None, None, None, None, 9999999
    best_stroke_ind = None
    best_stroke_bool_map = None

    diff = np.mean(np.abs(curr_canvas - target), axis=2) * (255. - np.mean(np.abs(color[None,None,:] - target), axis=2))
    # diff = (255. - np.mean(np.abs(color[None,None,:] - target), axis=2)) # best in practice
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
    diff = diff/diff.sum() # turn to probability distribution

    # plt.imshow(diff)
    # plt.colorbar()
    # plt.show()
    target = target.astype(np.float32)
    for x_y_attempt in range(x_y_attempts): # Try a few random x/y's
        #x, y = np.random.randint(target.shape[1]), np.random.randint(target.shape[0])

        # ind = np.random.randint(len(good_x_inds))
        #x, y = good_x_inds[ind], good_y_inds[ind]  
        
        y, x = np.unravel_index(np.random.choice(len(diff.flatten()), p=diff.flatten()), diff.shape)


        for stroke_ind in range(len(strokes)):
            stroke = strokes[stroke_ind]
            for rot in range(0, 360, 15):
                #print(curr_canvas.max(), stroke.max(), x, y, color)
                candidate_canvas, stroke_bool_map = apply_stroke(curr_canvas.copy(), stroke, stroke_ind,
                    color, x, y, rot)

                
                # if weight is not None:
                #     loss = np.mean(weight[:,:,None] * loss_fcn(target, candidate_canvas))
                # else:
                loss = np.mean(loss_fcn(target, candidate_canvas.astype(np.float32)))

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
    
    # show_img(best_canvas/255.)
    return 1.*best_x/curr_canvas.shape[1], 1 - 1.*best_y/curr_canvas.shape[0],\
            best_stroke_ind, best_rot, best_canvas/255., best_loss, diff, best_stroke_bool_map

import matplotlib
import matplotlib.pyplot as plt

def show_img(img):
    # Display at actual size: https://stackoverflow.com/questions/60144693/show-image-in-its-original-resolution-in-jupyter-notebook
    # Acquire default dots per inch value of matplotlib
    dpi = matplotlib.rcParams['figure.dpi']
    # Determine the figures size in inches to fit your image
    height, width = img.shape[0], img.shape[1]
    figsize = width / float(dpi), height / float(dpi)

    plt.figure(figsize=figsize)
    plt.title("Proposed Next Stroke")
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
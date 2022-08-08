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

from options import Options
from tensorboard import TensorBoard

# from torch_painting_models import *
from torch_painting_models_continuous import *
from style_loss import compute_style_loss


from clip_loss import clip_conv_loss, clip_model, clip_text_loss
import clip
import kornia as K

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

normalize_img = transforms.Compose([
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

loss_l1 = torch.nn.L1Loss()


# Utilities


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




# Loading Strokes

def load_brush_strokes(opt, scale_factor=4):
    with gzip.open(os.path.join(opt.cache_dir, 'strokes_centered.npy'), 'rb') as f:
        strokes = pickle.load(f, encoding="latin1")

    strokes_processed = []
    for s in strokes:
        s = cv2.resize(s, (int(s.shape[1]/scale_factor), int(s.shape[0]/scale_factor)))

        strokes_processed.append(torch.from_numpy(s).float().to(device) / 255.)
    return strokes_processed

strokes_small = None#load_brush_strokes(scale_factor=3)
strokes_full = None#load_brush_strokes(scale_factor=1)


writer = None
target = None
local_it = 0 

def log_progress(painting, force_log=False):
    global local_it
    local_it +=1
    if (local_it %10==0) or force_log:
        with torch.no_grad():
            np_painting = painting(strokes=strokes_small, use_alpha=False).detach().cpu().numpy()[0].transpose(1,2,0)
            opt.writer.add_image('images/planasdfasdf', np.clip(np_painting, a_min=0, a_max=1), local_it)
            if target is not None:
                opt.writer.add_scalar('loss/plan_all_strokes_local', loss_fcn(painting(strokes=strokes_small), target).item(), local_it)

def log_painting(painting, step, name='images/plan_all_strokes'):
    np_painting = painting(strokes=strokes_small).detach().cpu().numpy()[0].transpose(1,2,0)
    opt.writer.add_image(name, np.clip(np_painting, a_min=0, a_max=1), step)

def sort_brush_strokes_by_color(painting, bin_size=3000):
    with torch.no_grad():
        brush_strokes = [bs for bs in painting.brush_strokes]
        for j in range(0,len(brush_strokes), bin_size):
            brush_strokes[j:j+bin_size] = sorted(brush_strokes[j:j+bin_size], 
                key=lambda x : x.color_transform.mean(), reverse=True)
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
        c = cv2.cvtColor(c, cv2.COLOR_RGB2Lab)
        #print('c', c.shape)

        
        dist = colour.delta_E(dc, c)
        #print(dist.shape)
        argmin = np.argmin(dist)

        return discrete_colors[argmin].clone()


def loss_fcn(painting, target, use_l2_loss=False, use_clip_loss=False, use_style_loss=False):
    loss = 0 

    #if use_l2_loss:
    diff = (painting[:,:3] - target)**2

    loss += diff.mean()

    if use_clip_loss:
        cl = clip_conv_loss(painting, target) #* 0.5
        #opt.writer.add_scalar('loss/content_loss', cl.item(), local_it)
        loss = cl

    if use_style_loss:
        sl = compute_style_loss(painting, target) * .5
        #opt.writer.add_scalar('loss/style_loss', sl.item(), local_it)
        loss += sl

        # loss += torch.nn.L1Loss()(K.filters.canny(painting[:,:3])[0], K.filters.canny(target)[0])
    return loss



def create_canvas_cache(painting):
    # Get map of brush stroke index to what the canvas looks like after that canvas
    future_canvas_cache = {}
    for i in range(len(painting.brush_strokes)-1, 0, -1):
        with torch.no_grad():
            mid_point_canvas = torch.zeros((1,4,strokes_small[0].shape[0],strokes_small[0].shape[1])).to(device)
            #for brush_stroke in painting.brush_strokes[i:]:
            for j in range(i,len(painting.brush_strokes),1):
                brush_stroke = painting.brush_strokes[j]
                single_stroke = brush_stroke(strokes_small)
                if j in future_canvas_cache.keys():
                    mid_point_canvas = mid_point_canvas * (1 - future_canvas_cache[j][:,3:]) + future_canvas_cache[j][:,3:] * future_canvas_cache[j]
                    break
                else:
                    mid_point_canvas = mid_point_canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
            future_canvas_cache[i] = mid_point_canvas
    return future_canvas_cache

def purge_extraneous_brush_strokes(painting, target):
    relaxed_brush_strokes = []

    future_canvas_cache = create_canvas_cache(painting)

    canvas_before = T.Resize(size=(strokes_small[0].shape[0],strokes_small[0].shape[1]))(painting.background_img.detach())
    for i in (range(len(painting.brush_strokes))):
        with torch.no_grad():
            canvas_after = torch.zeros((1,4,strokes_small[0].shape[0],strokes_small[0].shape[1])).to(device)
            for j in range(i+1,len(painting.brush_strokes),1):
                brush_stroke = painting.brush_strokes[j]
                single_stroke = brush_stroke(strokes_small)
                single_stroke[:,3][single_stroke[:,3] > 0.2] = 1.
                if j in future_canvas_cache.keys():
                    canvas_after = canvas_after * (1 - future_canvas_cache[j][:,3:]) + future_canvas_cache[j][:,3:] * future_canvas_cache[j]
                    break
                else:
                    canvas_after = canvas_after * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke

        brush_stroke = painting.brush_strokes[i]
        single_stroke = brush_stroke(strokes_small).detach()
        single_stroke[:,3][single_stroke[:,3] > 0.2] = 1.

        canvas_without_stroke = canvas_before * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after
        canvas_with_stroke = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
        canvas_with_stroke = canvas_with_stroke * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after
        # show_img(canvas_without_stroke)
        # show_img(canvas_with_stroke)
        loss_without_stroke = nn.L1Loss()(canvas_without_stroke[:,:3], target)
        loss_with_stroke = nn.L1Loss()(canvas_with_stroke[:,:3], target)
        # with torch.no_grad():
        #     loss_without_stroke = loss_fcn(canvas_without_stroke[:,:3], target, use_clip_loss=True)
        #     loss_with_stroke = loss_fcn(canvas_with_stroke[:,:3], target, use_clip_loss=True)

        if loss_with_stroke + 1e-6 < loss_without_stroke:
            # Keep the stroke
            canvas_before = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
            relaxed_brush_strokes.append(brush_stroke)

    new_painting = copy.deepcopy(painting)
    new_painting.brush_strokes = nn.ModuleList(relaxed_brush_strokes)
    return new_painting

def purge_buried_brush_strokes(painting):
    relaxed_brush_strokes = []

    future_canvas_cache = create_canvas_cache(painting)

    canvas_before = T.Resize(size=(strokes_small[0].shape[0],strokes_small[0].shape[1]))(painting.background_img.detach())
    for i in (range(len(painting.brush_strokes))):
        with torch.no_grad():
            canvas_after = torch.zeros((1,4,strokes_small[0].shape[0],strokes_small[0].shape[1])).to(device)
            for j in range(i+1,len(painting.brush_strokes),1):
                brush_stroke = painting.brush_strokes[j]
                single_stroke = brush_stroke(strokes_small)
                single_stroke[:,3][single_stroke[:,3] > 0.2] = 1.
                if j in future_canvas_cache.keys():
                    canvas_after = canvas_after * (1 - future_canvas_cache[j][:,3:]) + future_canvas_cache[j][:,3:] * future_canvas_cache[j]
                    break
                else:
                    canvas_after = canvas_after * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke

        brush_stroke = painting.brush_strokes[i]
        single_stroke = brush_stroke(strokes_small).detach()
        single_stroke[:,3][single_stroke[:,3] > 0.2] = 1.

        canvas_without_stroke = canvas_before * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after
        canvas_with_stroke = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
        canvas_with_stroke = canvas_with_stroke * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after
        # show_img(canvas_without_stroke)
        # show_img(canvas_with_stroke)

        #if torch.mean(torch.abs(canvas_without_stroke[:,:3] - canvas_with_stroke[:,:3])) > 1e-5:
        # print(canvas_with_stroke.shape, canvas_without_stroke.shape)
        if torch.sum(torch.isclose(canvas_with_stroke[:,:3], canvas_without_stroke[:,:3])) > 10:
            # Keep the stroke
            canvas_before = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
            relaxed_brush_strokes.append(brush_stroke)

    new_painting = copy.deepcopy(painting)
    new_painting.brush_strokes = nn.ModuleList(relaxed_brush_strokes)
    return new_painting


def plan_all_strokes_text(opt, optim_iter=1000, num_strokes=700, num_augs=80, num_passes=1):
    global strokes_small, strokes_full, target
    strokes_small = load_brush_strokes(opt, scale_factor=10)
    strokes_full = load_brush_strokes(opt, scale_factor=1)

    with torch.no_grad():
        text_features = clip_model.encode_text(clip.tokenize(opt.prompt).to(device))
    
    target = load_img(opt.target,h=strokes_small[0].shape[0], w=strokes_small[0].shape[1]).to(device)/255.

    colors = get_colors(cv2.resize(cv2.imread(opt.target)[:,:,::-1], (256, 256)), n_colors=opt.n_colors)
    with open(os.path.join(opt.cache_dir, 'colors.npy'), 'rb') as f:
        colors = np.load(f)
    colors = (torch.from_numpy(np.array(colors)) / 255.).float().to(device)
    # print(strokes_small[0].shape)


    # Get the background of painting to be the current canvas
    current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg')).to(device)/255.

    current_canvas = current_canvas#*0 + 1.
    painting = Painting(0, background_img=current_canvas, unique_strokes=len(strokes_small)).to(device)

    for i in range(num_passes):
        # Add a brush strokes
        canvas = painting(strokes=strokes_small)

        gridded_brush_strokes = []

        h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]

        xys = [(x,y) for x in torch.linspace(-.99,.99,20) for y in torch.linspace(-.99,.99,20)]

        
        random.shuffle(xys)
        for x,y in tqdm(xys):
            opt_params = { # Find the optimal parameters
                'brush_stroke':None, 'canvas':None, 'loss':9999999, 'stroke_ind':None,
            }
            # solve for stroke type, color, and rotation
            for x_y_attempt in range(20):
                # Random brush stroke
                color = target[:,:3,int((y+1)/2*target.shape[2]), int((x+1)/2*target.shape[3])][0]
                brush_stroke = BrushStroke(np.random.randint(len(strokes_small)), 
                    xt=x,
                    yt=y,
                    color=color.detach().clone())
                    # color=colors[np.random.randint(len(colors))].clone()).to(device)
                        
                single_stroke = brush_stroke(strokes_small)
                # single_stroke[:,3][single_stroke[:,3] > 0.5] = 1. # opaque
                canvas_candidate = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
                
                with torch.no_grad():
                    #loss = loss_fcn(canvas_candidate, target,  use_clip_loss=False).item()
                    diff = torch.abs(canvas_candidate - target)
                    diff[diff>0.05] = 1.
                    loss = diff.mean()

                #loss = loss_fcn(canvas_candidate, target, use_clip_loss=False)
                if loss < opt_params['loss']:
                    opt_params['canvas'] = canvas_candidate
                    opt_params['loss'] = loss
                    opt_params['brush_stroke'] = brush_stroke
            canvas = opt_params['canvas']
            gridded_brush_strokes.append(opt_params['brush_stroke'])

            # painting = Painting(0, background_img=current_canvas, brush_strokes=gridded_brush_strokes).to(device)
            # log_progress(painting)
            if len(gridded_brush_strokes) % 50 == 0:
                np_painting = canvas.detach().cpu().numpy()[0].transpose(1,2,0)
                opt.writer.add_image('images/grid_add', np.clip(np_painting, a_min=0, a_max=1), len(gridded_brush_strokes))

        painting = Painting(0, background_img=current_canvas, 
            brush_strokes=[bs for bs in painting.brush_strokes] + gridded_brush_strokes).to(device)
        discretize_colors(painting, colors)
        painting = sort_brush_strokes_by_color(painting)

        # Optimize all brush strokes
        print('Optimizing all {} brush strokes'.format(str(len(painting.brush_strokes))))
        optim = torch.optim.RMSprop(painting.parameters(), lr=5e-3)# * (len(painting.brush_strokes)/100))
        for j in tqdm(range(optim_iter)):
            optim.zero_grad()
            p = painting(strokes=strokes_small)
            loss = clip_text_loss(p, text_features, num_augs)
            sl = compute_style_loss(p[:,:3], target) * .5
            loss += sl
            loss.backward()
            # for bs in painting.brush_strokes:
            #     bs.color_transform.grad.data *= 0. # Don't change the color because CLIP sucks at color
            optim.step()

            if j % 30 == 0 and j > (100):
                discretize_colors(painting, colors)
            log_progress(painting)

    discretize_colors(painting, colors)
    log_progress(painting)
    # show_img(painting(strokes=strokes_full))
    return painting

def segment_strokes_by_size(strokes, n_segments):
    ''' Returne the indices of the strokes binned by size in descending order '''
    print(strokes[0].shape)
    stroke_sizes = -1. * np.array([s[:,:,-1].sum().detach().cpu().numpy() for s in strokes])
    sizes_ranked = stroke_sizes.argsort().argsort().tolist()

    n_strokes = len(strokes)
    r = np.arange(n_strokes)

    stroke_inds = []
    for i in range(n_segments):
        seg_ranks = r[i*int(n_strokes/n_segments):(i+1)*int(n_strokes/n_segments)]
        if i == n_segments - 1:
            seg_ranks = r[i*int(n_strokes/n_segments):]

        strokes_in_seg = [sizes_ranked.index(sr) for sr in seg_ranks]
        stroke_inds.append(strokes_in_seg)
    print(stroke_inds)
    return stroke_inds

# def plan_all_strokes_grid(opt, optim_iter=150, num_strokes_x=25, num_strokes_y=22, 
#             x_y_attempts=200, num_passes=3):
#     global strokes_small, strokes_full, target
#     strokes_small = load_brush_strokes(opt, scale_factor=4)
#     strokes_full = load_brush_strokes(opt, scale_factor=4)

#     strokes_by_size = segment_strokes_by_size(strokes_small, num_passes)
    
#     h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]

#     # target = load_img(opt.target,
#     #     h=strokes_small[0].shape[0], w=strokes_small[0].shape[1]).to(device)/255.
#     target = load_img(os.path.join(opt.cache_dir, 'target_discrete.jpg'),
#         h=strokes_small[0].shape[0], w=strokes_small[0].shape[1]).to(device)/255.

#     opt.writer.add_image('target/target', np.clip(target.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), local_it)

#     # colors = get_colors(cv2.resize(cv2.imread(opt.target)[:,:,::-1], (256, 256)), n_colors=opt.n_colors)
#     with open(os.path.join(opt.cache_dir, 'colors.npy'), 'rb') as f:
#         colors = np.load(f)
#     colors = (torch.from_numpy(np.array(colors)) / 255.).to(device)


#     # Get the background of painting to be the current canvas
#     current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg')).to(device)/255.
#     painting = Painting(0, background_img=current_canvas, 
#         unique_strokes=len(strokes_small)).to(device)
#     layer_background = painting(strokes=strokes_small).clone()

#     total_strokes = []

#     target_lab = K.color.lab.rgb_to_lab(target)
#     for i in range(num_passes):
#         if opt.just_fine and i != num_passes-1: continue
#         canvas = layer_background
#         painting = Painting(0, background_img=canvas, 
#             unique_strokes=len(strokes_small)).to(device)

#         layer_brush_strokes = []

#         xys = [(x,y) for x in torch.linspace(-.99,.99,num_strokes_x) for y in torch.linspace(-.99,.99,num_strokes_y)]
#         # xys = [(x,y) for x in torch.linspace(-.99,.99,num_strokes_x) for y in torch.linspace(-.99,0,num_strokes_y)]
#         # xys = [(x,y) for x in torch.linspace(-.3,.99,num_strokes_x) for y in torch.linspace(-.99,.6,num_strokes_y)]
#         k = 0
#         random.shuffle(xys)
#         for x,y in tqdm(xys):
#             opt_params = { # Find the optimal parameters
#                 'brush_stroke':None, 'canvas':None, 'loss':9999999, 'stroke_ind':None,
#             }
#             # solve for stroke type, color, and rotation
#             og_correct_pix = torch.sum(torch.isclose(K.color.lab.rgb_to_lab(canvas[:,:3])/127., target_lab/127., atol=1e-2).float())
#             og_incorrect_pix = torch.sum(1-torch.isclose(K.color.lab.rgb_to_lab(canvas[:,:3])/127., target_lab/127., atol=1e-2).float())
#             for x_y_attempt in range(x_y_attempts):
#                 # Random brush stroke
#                 color = target[:,:3,int((y+1)/2*target.shape[2]), int((x+1)/2*target.shape[3])][0]
#                 brush_stroke = BrushStroke(random.choice(strokes_by_size[i]),
#                     xt=x,
#                     yt=y,
#                     a=(np.random.randint(20)-10)/10*3.14,
#                     color=color.detach().clone())
#                     # color=colors[np.random.randint(len(colors))].clone()).to(device)
                        
#                 single_stroke = brush_stroke(strokes_small)
#                 # print(canvas.shape, single_stroke.shape)
#                 # single_stroke[:,3][single_stroke[:,3] > 0.5] = 1. # opaque
#                 canvas_candidate = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
                
#                 with torch.no_grad():
#                     canvas_candidate_lab = K.color.lab.rgb_to_lab(canvas_candidate)
#                     # loss = torch.mean((canvas_candidate_lab - target_lab)**2)

#                     correct_pix = torch.sum(torch.isclose(canvas_candidate_lab[:,:3]/127., target_lab/127., atol=1e-2).float())
#                     incorrect_pix = torch.sum(1-torch.isclose(canvas_candidate_lab[:,:3]/127., target_lab/127., atol=1e-2).float())
#                     loss = -1. * (correct_pix - og_correct_pix)
#                     loss += (incorrect_pix - og_incorrect_pix) * (i+2)

#                 #loss = loss_fcn(canvas_candidate, target, use_clip_loss=False)
#                 if loss < opt_params['loss']:
#                     opt_params['canvas'] = canvas_candidate
#                     opt_params['loss'] = loss
#                     opt_params['brush_stroke'] = brush_stroke
#             canvas = opt_params['canvas']
#             layer_brush_strokes.append(opt_params['brush_stroke'])

#             # Do a lil optimization on the most recent strokes
#             if k % 30 == 0:
#                 strokes_to_optimize = layer_brush_strokes[-100:]
#                 older_strokes = layer_brush_strokes[:-100]
#                 back_p = Painting(0, background_img=layer_background, 
#                     brush_strokes=older_strokes).to(device)
#                 background_img = back_p(strokes=strokes_small, use_alpha=False)
#                 p = Painting(0, background_img=background_img, 
#                     brush_strokes=strokes_to_optimize).to(device)
#                 optim = torch.optim.Adam(p.parameters(), lr=1e-2)
#                 for j in (range(10)):
#                     optim.zero_grad()
#                     loss = loss_fcn(p(strokes=strokes_small, use_alpha=False), target,  use_clip_loss=False, use_style_loss=False)
#                     loss.backward()
#                     for bs in p.brush_strokes:
#                         bs.color_transform.grad.data *= 0. # Don't change the color because CLIP sucks at color
#                     optim.step()

#                 with torch.no_grad():
#                     layer_brush_strokes = older_strokes
#                     layer_brush_strokes += [bs for bs in p.brush_strokes]
#                     p = Painting(0, background_img=layer_background, 
#                         brush_strokes=layer_brush_strokes).to(device)
                

#                 # p = Painting(0, background_img=current_canvas, 
#                 #     brush_strokes=layer_brush_strokes).to(device)
#                 # optim = torch.optim.Adam(p.parameters(), lr=1e-2)# * (len(painting.brush_strokes)/100))
#                 # for j in (range(10)):
#                 #     optim.zero_grad()
#                 #     loss = loss_fcn(p(strokes=strokes_small, use_alpha=False), target,  use_clip_loss=False, use_style_loss=False)
#                 #     # loss = loss_fcn(p(strokes=strokes_small, use_alpha=False), target,  use_clip_loss=True, use_style_loss=False)
#                 #     loss.backward()
#                 #     # for bs in p.brush_strokes:
#                 #     #     bs.color_transform.grad.data *= 0. # Don't change the color because CLIP sucks at color
#                 #     optim.step()

#                 # with torch.no_grad():
#                 #     layer_brush_strokes = [bs for bs in p.brush_strokes]
#                 #     p = Painting(0, background_img=current_canvas, 
#                 #         brush_strokes=[bs for bs in painting.brush_strokes] + layer_brush_strokes).to(device)
#                 with torch.no_grad():
#                     discretize_colors(p, colors)
#                     p = sort_brush_strokes_by_color(p)
#                     # n_strokes = len(p.brush_strokes)
#                     # p = purge_buried_brush_strokes(p)
#                     # if len(p.brush_strokes) != n_strokes:
#                     #     print('removed', n_strokes - len(p.brush_strokes), 'brush strokes')
#                     n_strokes = len(p.brush_strokes)
#                     p = purge_extraneous_brush_strokes(p, target)
#                     if len(p.brush_strokes) != n_strokes:
#                         print('removed', n_strokes - len(p.brush_strokes), 'brush strokes that did not help')
                
                
#                     layer_brush_strokes = [bs for bs in p.brush_strokes]
#                     canvas = p(strokes=strokes_small, use_alpha=False)

#             # painting = Painting(0, background_img=current_canvas, brush_strokes=layer_brush_strokes).to(device)
#             # log_progress(painting)
#             if k % 1 == 0:
#                 np_painting = canvas.detach().cpu().numpy()[0].transpose(1,2,0)
#                 opt.writer.add_image('images/grid_add_layer{}'.format(i), 
#                     np.clip(np_painting, a_min=0, a_max=1), k)
#             k += 1
#         painting = Painting(0, background_img=layer_background, 
#             brush_strokes=layer_brush_strokes).to(device)
#         discretize_colors(painting, colors)
#         log_progress(painting)

        
#         # Optimize all brush strokes
#         print('Optimizing all {} brush strokes'.format(str(len(painting.brush_strokes))))
#         optim = torch.optim.Adam(painting.parameters(), lr=1e-2)# * (len(painting.brush_strokes)/100))
#         for j in tqdm(range(optim_iter)):
#             optim.zero_grad()
#             p = painting(strokes=strokes_small, use_alpha=False)
#             loss = 0
#             loss += loss_fcn(p, target,  use_clip_loss=True, use_style_loss=False)
#             loss.backward()
#             for bs in painting.brush_strokes:
#                 bs.color_transform.grad.data *= 0. # Don't change the color because CLIP sucks at color
#             optim.step()
#             log_progress(painting)

#             optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * 0.99

#             if j % 10 == 0 and (j > .25*optim_iter):
#                 discretize_colors(painting, colors)
#                 painting = sort_brush_strokes_by_color(painting)
#                 optim = torch.optim.Adam(painting.parameters(), lr=optim.param_groups[0]['lr'])

#         discretize_colors(painting, colors)
#         painting = sort_brush_strokes_by_color(painting)

#         total_strokes += [bs for bs in painting.brush_strokes]
#         layer_background = painting(strokes=strokes_small).detach().clone()

#     painting = Painting(0, background_img=current_canvas, 
#             brush_strokes=total_strokes).to(device)
#     return painting

def plan_all_strokes_grid_continuous(opt, optim_iter=100, num_strokes_x=15, num_strokes_y=15, 
            x_y_attempts=1, num_passes=1):
    global strokes_small, strokes_full, target
    strokes_full = load_brush_strokes(opt, scale_factor=1)
    # print(strokes_full[0].shape)
    scale_factor = strokes_full[0].shape[0] / opt.max_height
    strokes_small = load_brush_strokes(opt, scale_factor=scale_factor)
    strokes_full = load_brush_strokes(opt, scale_factor=scale_factor)

    strokes_by_size = segment_strokes_by_size(strokes_small, num_passes)
    
    h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]

    # target = load_img(opt.target,
    #     h=strokes_small[0].shape[0], w=strokes_small[0].shape[1]).to(device)/255.
    target = load_img(os.path.join(opt.cache_dir, 'target_discrete.jpg'),
        h=strokes_small[0].shape[0], w=strokes_small[0].shape[1]).to(device)/255.

    opt.writer.add_image('target/target', np.clip(target.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1), local_it)

    # colors = get_colors(cv2.resize(cv2.imread(opt.target)[:,:,::-1], (256, 256)), n_colors=opt.n_colors)
    with open(os.path.join(opt.cache_dir, 'colors.npy'), 'rb') as f:
        colors = np.load(f)
    colors = (torch.from_numpy(np.array(colors)) / 255.).to(device)


    # Get the background of painting to be the current canvas
    current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg')).to(device)/255.
    painting = Painting(0, background_img=current_canvas, 
        unique_strokes=len(strokes_small)).to(device)
    layer_background = painting(strokes=strokes_small).clone()

    total_strokes = []

    target_lab = K.color.lab.rgb_to_lab(target)
    for i in range(num_passes):
        if opt.just_fine and i != num_passes-1: continue
        canvas = layer_background
        painting = Painting(0, background_img=canvas, 
            unique_strokes=len(strokes_small)).to(device)

        layer_brush_strokes = []

        xys = [(x,y) for x in torch.linspace(-.99,.99,num_strokes_x) for y in torch.linspace(-.99,.99,num_strokes_y)]
        # xys = [(x,y) for x in torch.linspace(-.99,.99,num_strokes_x) for y in torch.linspace(-.99,0,num_strokes_y)]
        # xys = [(x,y) for x in torch.linspace(-.3,.99,num_strokes_x) for y in torch.linspace(-.99,.6,num_strokes_y)]
        k = 0
        random.shuffle(xys)
        for x,y in tqdm(xys):
            opt_params = { # Find the optimal parameters
                'brush_stroke':None, 'canvas':None, 'loss':9999999, 'stroke_ind':None,
            }
            # solve for stroke type, color, and rotation
            og_correct_pix = torch.sum(torch.isclose(K.color.lab.rgb_to_lab(canvas[:,:3])/127., target_lab/127., atol=1e-2).float())
            og_incorrect_pix = torch.sum(1-torch.isclose(K.color.lab.rgb_to_lab(canvas[:,:3])/127., target_lab/127., atol=1e-2).float())
            for x_y_attempt in range(x_y_attempts):
                # Random brush stroke
                color = target[:,:3,int((y+1)/2*target.shape[2]), int((x+1)/2*target.shape[3])][0]
                brush_stroke = BrushStroke(random.choice(strokes_by_size[i]),
                    xt=x,
                    yt=y,
                    a=(np.random.randint(20)-10)/10*3.14,
                    color=color.detach().clone())
                    # color=colors[np.random.randint(len(colors))].clone()).to(device)
                        
                single_stroke = brush_stroke(strokes_small)
                # print(canvas.shape, single_stroke.shape)
                # single_stroke[:,3][single_stroke[:,3] > 0.5] = 1. # opaque
                canvas_candidate = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
                
                with torch.no_grad():
                    canvas_candidate_lab = K.color.lab.rgb_to_lab(canvas_candidate)
                    # loss = torch.mean((canvas_candidate_lab - target_lab)**2)

                    correct_pix = torch.sum(torch.isclose(canvas_candidate_lab[:,:3]/127., target_lab/127., atol=1e-2).float())
                    incorrect_pix = torch.sum(1-torch.isclose(canvas_candidate_lab[:,:3]/127., target_lab/127., atol=1e-2).float())
                    loss = -1. * (correct_pix - og_correct_pix)
                    loss += (incorrect_pix - og_incorrect_pix) * (i+2)

                #loss = loss_fcn(canvas_candidate, target, use_clip_loss=False)
                if loss < opt_params['loss']:
                    opt_params['canvas'] = canvas_candidate
                    opt_params['loss'] = loss
                    opt_params['brush_stroke'] = brush_stroke
            canvas = opt_params['canvas']
            layer_brush_strokes.append(opt_params['brush_stroke'])

            if k % 1 == 0:
                np_painting = canvas.detach().cpu().numpy()[0].transpose(1,2,0)
                opt.writer.add_image('images/grid_add_layer{}'.format(i), 
                    np.clip(np_painting, a_min=0, a_max=1), k)
            k += 1
        painting = Painting(0, background_img=layer_background, 
            brush_strokes=layer_brush_strokes).to(device)
        discretize_colors(painting, colors)
        log_progress(painting)

        
        # Optimize all brush strokes
        print('Optimizing all {} brush strokes'.format(str(len(painting.brush_strokes))))
        optim = torch.optim.Adam(painting.parameters(), lr=1e-2)# * (len(painting.brush_strokes)/100))
        canvases = []
        for j in tqdm(range(optim_iter)):
            optim.zero_grad()
            p = painting(strokes=strokes_small, use_alpha=False)
            loss = 0
            loss += loss_fcn(p, target,  use_clip_loss=True, use_style_loss=False)
            loss += loss_fcn(p, target,  use_clip_loss=False, use_style_loss=False)
            loss.backward()
            # for bs in painting.brush_strokes:
            #     bs.color_transform.grad.data *= 0. # Don't change the color because CLIP sucks at color
            optim.step()
            painting.validate()
            log_progress(painting)#, force_log=True)

            optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * 0.99

            if j % 10 == 0 and (j > .25*optim_iter):
                discretize_colors(painting, colors)
                painting = sort_brush_strokes_by_color(painting)
                optim = torch.optim.Adam(painting.parameters(), lr=optim.param_groups[0]['lr'])
            canvases.append(np.clip(p.detach().cpu().numpy()[0].transpose(1,2,0), a_min=0, a_max=1))
        discretize_colors(painting, colors)
        painting = sort_brush_strokes_by_color(painting)

        total_strokes += [bs for bs in painting.brush_strokes]
        layer_background = painting(strokes=strokes_small).detach().clone()

        from paint_utils import to_video
        import time
        to_video(canvases, fn='/home/frida/Videos/frida/plan_optimization_canvases{}.mp4'.format(str(time.time())))

    painting = Painting(0, background_img=current_canvas, 
            brush_strokes=total_strokes).to(device)
    return painting



if __name__ == '__main__':
    global opt
    opt = Options()
    opt.gather_options()


    b = './painting'
    all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
    tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
    if '_planner' not in tensorboard_dir:
        tensorboard_dir += '_planner'

    writer = TensorBoard(tensorboard_dir)
    opt.writer = writer


    if opt.prompt is not None:
        painting = plan_all_strokes_text(opt)
    else:
        if opt.discrete:
            painting = plan_all_strokes_grid(opt)
        else:
            painting = plan_all_strokes_grid_continuous(opt)

    # Export the strokes
    f = open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), "w")
    f.write(painting.to_csv())
    f.close()
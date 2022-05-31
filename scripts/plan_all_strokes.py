import pickle
import numpy as np
import torch
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

from torch_painting_models import *

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# Utilities


def load_img_internet(url, h=None, w=None):
    response = requests.get(url).content
    im = Image.open(io.BytesIO(response))
    im = np.array(im)
    if im.shape[1] > max_size:
        fact = im.shape[1] / max_size
    im = cv2.resize(im, (w,h)) if h is not None and w is not None else im
    im = torch.from_numpy(im)
    im = im.permute(2,0,1)
    #print(im.shape)
    return im.unsqueeze(0).float()

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



# CLIP and LPIPS

# import clip
# clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

from torchvision import transforms

# LPIPS
# loss_fn_alex = lpips.LPIPS(net='alex').to(device)
# lpips_transform = transforms.Compose([transforms.Resize((64,64))])

# import ttools.modules
# perception_loss = ttools.modules.LPIPS().to(device)

def get_image_augmentation(use_normalized_clip):
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return augment_trans

# Loading Strokes

# def load_brush_strokes(opt, scale_factor=4):
#     # Strokes are the individual strokes pasted onto the middle of a full sized canvas
#     with open(os.path.join(opt.cache_dir, 'strokes_centered.npy'), 'rb') as f:
#         strokes = pickle.load(f, encoding="latin1")


#     #h, w = int(1024/scale_factor), int((10/8)*1024/scale_factor)
#     h, w = int(opt.CANVAS_HEIGHT_PIX/scale_factor), int(opt.CANVAS_WIDTH_PIX/scale_factor)
#     canvas = torch.zeros((h, w, 3))
#     # strokes = [torch.from_numpy(s) for s in strokes]
#     strokes_processed = []
#     for s in strokes:
#         s = s[:,:,3] # Just the alpha channel is necessary

#         # Scale it for speed
#         s = cv2.resize(s, (int(s.shape[1]/scale_factor), int(s.shape[0]/scale_factor)))

#         # Pad the stroke so its on an image the size of a canvas and the start of the stroke is centered
#         padX = int((w - s.shape[1])/2)
#         padY = int((h - s.shape[0])/2)
#         s = np.pad(s, [(padY,padY), (padX,padX)], 'constant')
#         t = np.zeros((s.shape[0], s.shape[1], 4), dtype=np.uint8)

#         t[:,:,3] = s.astype(np.uint8)
#         t[:,:,2] = 200 # Some color for fun
#         t[:,:,1] = 120

#         strokes_processed.append(torch.from_numpy(t).float().to(device) / 255.)

#     for i in range(len(strokes_processed)):
#         # show_img(strokes[i])
#         show_img(strokes_processed[i])
#         # show_img(strokes_processed[i][:,:,-1])
#         # print(strokes_processed[i][:,:,0].max())
#     strokes = strokes_processed


def load_brush_strokes(opt, scale_factor=4):
    with open(os.path.join(opt.cache_dir, 'strokes_centered.npy'), 'rb') as f:
        strokes = pickle.load(f, encoding="latin1")

    # h, w = int(strokes[0].shape[0]/scale_factor), int(strokes[0].shape[1]/scale_factor)
    # canvas = torch.zeros((h, w, 3))

    strokes_processed = []
    for s in strokes:
        s = cv2.resize(s, (int(s.shape[1]/scale_factor), int(s.shape[0]/scale_factor)))

        # padX = int((w - s.shape[1])/2)
        # padY = int((h - s.shape[0])/2)

        # # In case of odd numbers
        # xtra_x, xtra_y = w - (padX*2+s.shape[1]), h - (padY*2+s.shape[0])
        # s = np.pad(s, [(padY,padY+xtra_y), (padX,padX+xtra_x)], 'constant')
        # t = np.zeros((s.shape[0], s.shape[1], 4), dtype=np.uint8)
        # t[:,:,3] = s.astype(np.uint8)
        # t[:,:,2] = 200 # Some color for fun
        # t[:,:,1] = 120

        strokes_processed.append(torch.from_numpy(s).float().to(device) / 255.)
    return strokes_processed

strokes_small = None#load_brush_strokes(scale_factor=3)
strokes_full = None#load_brush_strokes(scale_factor=1)


# def next_stroke(painting, target, x_y_attempts=2):
#     # Plan a single brush stroke
#     with torch.no_grad():
#         l1_loss = nn.L1Loss()
#         canvas = painting#painting(strokes=strokes_small)
#         diff = torch.mean(torch.abs(canvas - target).unsqueeze(dim=0), dim=2)
#         #show_img(diff)
#         diff /= diff.sum() # Prob distribution

#         diff = diff.detach().cpu()[0][0].numpy()

#     opt_params = { # Find the optimal parameters
#         'x':None, 'y':None, 'rot':None,
#         'stroke':None, 'canvas':None, 'loss':9999999,
#         'stroke_ind':None, 'stroke_bool_map':None,
#     }

#     for x_y_attempt in range(x_y_attempts):
#         y, x = np.unravel_index(np.random.choice(len(diff.flatten()), p=diff.flatten()), diff.shape)
#         color = target[0,:,y,x]
#         h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]
#         y = torch.from_numpy(np.array(y))/h*2 - 1
#         x = torch.from_numpy(np.array(x))/w*2 - 1

#         for stroke_ind in range(len(strokes_small)):
#             # for rot in range(0, 360, 45):
#             #     brush_stroke = BrushStroke(stroke_ind, color=color, a=rot/(3.14*2), xt=x, yt=y).to(device)
#             #     single_stroke = brush_stroke(strokes=strokes_small)
#             #     #print(canvas.shape, single_stroke[:,3:].shape)
#             #     canvas_candidate = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]

#             #     #loss = loss_fcn(canvas_candidate, target) + nn.L1Loss()(canvas_candidate, target)
#             #     loss = l1_loss(canvas_candidate, target)

#             #     if loss < opt_params['loss']:
#             #         opt_params['x'], opt_params['y'] = x, y
#             #         opt_params['rot'] = rot
#             #         opt_params['canvas'] = canvas_candidate
#             #         opt_params['loss'] = loss
#             #         opt_params['stroke_ind'] = stroke_ind
#             #         opt_params['brush_stroke'] = brush_stroke
#             brush_stroke = BrushStroke(stroke_ind, color=color, a=None, xt=x, yt=y).to(device)
#             opt = torch.optim.Adam(brush_stroke.parameters(), lr=4e-2)
#             for brush_opt_iter in range(10):
#                 opt.zero_grad()
#                 single_stroke = brush_stroke(strokes=strokes_small)
#                 canvas_candidate = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
#                 loss = 0
#                 loss += nn.L1Loss()(canvas_candidate, target)
#                 #loss += clip_conv_loss(canvas_candidate, target).item()*0.2
#                 loss.backward()
#                 opt.step()
#             if loss < opt_params['loss']:
#                 opt_params['canvas'] = canvas_candidate
#                 opt_params['loss'] = loss
#                 opt_params['stroke_ind'] = stroke_ind
#                 opt_params['brush_stroke'] = brush_stroke
#         #print(opt_params['loss'])
#         return opt_params['brush_stroke'], opt_params['canvas']

# def purge_extraneous_strokes(painting, target, thresh=0):
#     # Remove strokes that don't help the L1 loss
#     l1_loss = nn.L1Loss()
#     with torch.no_grad():
#         canvas_with_stroke = painting(strokes=strokes_small)
#         i = 0
#         while i < len(painting.brush_strokes):
#             removed_stroke = painting.brush_strokes[i]
#             painting.brush_strokes = painting.brush_strokes[0:i] + painting.brush_strokes[i+1:]
#             canvas_without_stroke = painting(strokes=strokes_small)
#             loss_with_stroke = l1_loss(canvas_with_stroke, target)
#             loss_without_stroke = l1_loss(canvas_without_stroke, target)

#             if loss_with_stroke < loss_without_stroke:
#                 # Don't purge stroke. Put stroke back and move on
#                 painting.brush_strokes.insert(i, removed_stroke)
#                 i+=1
#             else:
#                 # Purge it
#                 canvas_with_stroke = canvas_without_stroke
#     return painting



#############################################################

def next_stroke(canvas, target, x_y_attempts=10):
    with torch.no_grad():
        diff = torch.mean(torch.abs(canvas[:,:3] - target).unsqueeze(dim=0), dim=2)
        diff /= diff.sum() # Prob distribution
        diff = diff.detach().cpu()[0][0].numpy()

    opt_params = { # Find the optimal parameters
        'brush_stroke':None, 'canvas':None, 'loss':9999999, 'stroke_ind':None,
    }

    for x_y_attempt in range(x_y_attempts):
        y, x = np.unravel_index(np.random.choice(len(diff.flatten()), p=diff.flatten()), diff.shape)
        color = target[0,:,y,x]
        h, w = strokes_small[0].shape[0], strokes_small[0].shape[1]
        y = torch.from_numpy(np.array(y))/h*2 - 1
        x = torch.from_numpy(np.array(x))/w*2 - 1

        for stroke_ind in range(len(strokes_small)):
            brush_stroke = BrushStroke(stroke_ind, color=color, a=None, xt=x, yt=y).to(device)
            # opt = torch.optim.Adam(brush_stroke.parameters(), lr=1e-2)
            # for brush_opt_iter in range(5):
            #     opt.zero_grad()
            #     single_stroke = brush_stroke(strokes_small)
            #     canvas_candidate = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
            #     loss = 0
            #     loss += nn.L1Loss()(canvas_candidate[:,:3], target)
            #     loss.backward()
            #     opt.step()
            single_stroke = brush_stroke(strokes_small)
            canvas_candidate = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
            loss = loss_fcn(canvas_candidate, target, use_clip_loss=False)
            if loss < opt_params['loss']:
                opt_params['canvas'] = canvas_candidate
                opt_params['loss'] = loss
                opt_params['stroke_ind'] = stroke_ind
                opt_params['brush_stroke'] = brush_stroke

    return opt_params['brush_stroke'], opt_params['canvas']

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

def relax(painting, target, batch_size=20):
    relaxed_brush_strokes = []

    future_canvas_cache = create_canvas_cache(painting)

    #print(painting.background_img.shape)
    canvas_before = T.Resize(size=(strokes_small[0].shape[0],strokes_small[0].shape[1]))(painting.background_img.detach())
    #print(canvas_before.shape)
    for i in tqdm(range(len(painting.brush_strokes))):
        with torch.no_grad():
            canvas_after = torch.zeros((1,4,strokes_small[0].shape[0],strokes_small[0].shape[1])).to(device)
            for j in range(i+1,len(painting.brush_strokes),1):
                brush_stroke = painting.brush_strokes[j]
                single_stroke = brush_stroke(strokes_small)
                if j in future_canvas_cache.keys():
                    canvas_after = canvas_after * (1 - future_canvas_cache[j][:,3:]) + future_canvas_cache[j][:,3:] * future_canvas_cache[j]
                    break
                else:
                    canvas_after = canvas_after * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke

        brush_stroke = painting.brush_strokes[i]
        best_stroke, best_loss = brush_stroke, 999999
        for stroke_ind in range(len(strokes_small)):
            brush_stroke.stroke_ind = stroke_ind
            opt = torch.optim.Adam(brush_stroke.parameters(), lr=5e-3)
            for it in range(10):
                opt.zero_grad()
                single_stroke = brush_stroke(strokes_small)
                canvas = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
                canvas = canvas * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after

                loss = loss_fcn(canvas, target, use_clip_loss=False)

                loss.backward()
                opt.step()

                if loss < best_loss:
                    best_stroke = copy.deepcopy(brush_stroke)
                    best_loss = loss

        brush_stroke = best_stroke
        single_stroke = brush_stroke(strokes_small).detach()

        canvas_before = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
        relaxed_brush_strokes.append(brush_stroke)

    relaxed_painting = copy.deepcopy(painting)
    relaxed_painting.brush_strokes = nn.ModuleList(relaxed_brush_strokes)
    return relaxed_painting

def purge_extraneous_brush_strokes(painting, target):
    relaxed_brush_strokes = []

    future_canvas_cache = create_canvas_cache(painting)

    canvas_before = T.Resize(size=(strokes_small[0].shape[0],strokes_small[0].shape[1]))(painting.background_img.detach())
    for i in tqdm(range(len(painting.brush_strokes))):
        with torch.no_grad():
            canvas_after = torch.zeros((1,4,strokes_small[0].shape[0],strokes_small[0].shape[1])).to(device)
            for j in range(i+1,len(painting.brush_strokes),1):
                brush_stroke = painting.brush_strokes[j]
                single_stroke = brush_stroke(strokes_small)
                if j in future_canvas_cache.keys():
                    canvas_after = canvas_after * (1 - future_canvas_cache[j][:,3:]) + future_canvas_cache[j][:,3:] * future_canvas_cache[j]
                    break
                else:
                    canvas_after = canvas_after * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke

        brush_stroke = painting.brush_strokes[i]
        single_stroke = brush_stroke(strokes_small).detach()

        canvas_without_stroke = canvas_before * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after
        canvas_with_stroke = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
        canvas_with_stroke = canvas_with_stroke * (1 - canvas_after[:,3:]) + canvas_after[:,3:] * canvas_after
        # show_img(canvas_without_stroke)
        # show_img(canvas_with_stroke)
        loss_without_stroke = nn.L1Loss()(canvas_without_stroke[:,:3], target)
        loss_with_stroke = nn.L1Loss()(canvas_with_stroke[:,:3], target)

        if loss_with_stroke + 0.001 < loss_without_stroke:
            # Keep the stroke
            canvas_before = canvas_before * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
            relaxed_brush_strokes.append(brush_stroke)

    new_painting = copy.deepcopy(painting)
    new_painting.brush_strokes = nn.ModuleList(relaxed_brush_strokes)
    return new_painting


def discretize_colors(painting, discrete_colors):
    pass
    # with torch.no_grad():
    #     for brush_stroke in painting.brush_strokes:
    #         brush_stroke.color_transform[:] = discretize_color(brush_stroke, discrete_colors)

def discretize_color(brush_stroke, discrete_colors):
    with torch.no_grad():
        color = brush_stroke.color_transform.detach()
        dist = torch.mean(torch.abs(discrete_colors - color[None,:])**2, dim=1)
        argmin = torch.argmin(dist)
        return discrete_colors[argmin]


from clip_loss import clip_conv_loss

loss_l1 = torch.nn.L1Loss()
def loss_fcn(painting, target, use_clip_loss=True):
    loss = 0 
    #return loss_l1(painting[:,:3], target)
    diff = torch.abs(painting[:,:3] - target)
    diff = diff**2
    loss += diff.mean()

    if use_clip_loss:
        loss += clip_conv_loss(painting, target)
    return loss


def plan_all_strokes(opt, optim_iter=200, num_strokes=50, num_passes=2):
    global strokes_small, strokes_full
    strokes_small = load_brush_strokes(opt, scale_factor=5)
    strokes_full = load_brush_strokes(opt, scale_factor=1)
    
    target = load_img(opt.target,h=strokes_small[0].shape[0], w=strokes_small[0].shape[1]).to(device)/255.

    colors = get_colors(cv2.resize(cv2.imread(opt.target)[:,:,::-1], (256, 256)), n_colors=opt.n_colors)
    colors = (torch.from_numpy(np.array(colors)) / 255.).to(device)
    # print(strokes_small[0].shape)


    # Get the background of painting to be the current canvas
    current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg')).to(device)/255.

    painting = Painting(0, background_img=current_canvas).to(device)


    for i in range(num_passes):
        print('\nPass{}'.format(str(i)))

        # Add a brush strokes
        print('Adding {} brush strokes'.format(str(num_strokes)))
        with torch.no_grad(): canvas = painting(strokes=strokes_small)
        for j in tqdm(range(num_strokes)):
            new_stroke, canvas = next_stroke(canvas.detach(), target)
            painting.brush_strokes.append(new_stroke)
            log_progress(painting)
        
        discretize_colors(painting, colors)

        # Optimize all brush strokes
        print('Optimizing all {} brush strokes'.format(str(len(painting.brush_strokes))))
        optim = torch.optim.Adam(painting.parameters(), lr=5e-3)
        for j in tqdm(range(optim_iter)):
            optim.zero_grad()
            p = painting(strokes=strokes_small)
            loss = 0
            loss += loss_fcn(p, target)
            loss.backward()
            optim.step()
            log_progress(painting)

        discretize_colors(painting, colors)

        # # Relax strokes: optimize each individually
        # print('Relaxing brush strokes')
        # painting = relax(painting, target)
        # with torch.no_grad():
        #     # show_img(painting(strokes=strokes_full))
        #     print(nn.L1Loss()(painting(strokes=strokes_small)[:,:3], target))

        # discretize_colors(painting, colors)

        # # Remove unnecessary brush strokes
        # print("Removing unnecessary brush strokes")
        # n_strokes_before = len(painting.brush_strokes)
        # painting = purge_extraneous_brush_strokes(painting, target)
        # print('Removed {} brush strokes. {} total now.'.format(str(len(painting.brush_strokes) - n_strokes_before), str(len(painting.brush_strokes))))
        # # with torch.no_grad():
        # #     show_img(painting(strokes=strokes_full))
        # #     print(nn.L1Loss()(painting(strokes=strokes_small)[:,:3], target))

        discretize_colors(painting, colors)

        opt.writer.add_scalar('loss/plan_all_strokes', loss_fcn(painting(strokes=strokes_small), target).item(), opt.global_it + i)

        log_painting(painting, opt.global_it + i)
    # show_img(painting(strokes=strokes_full))
    return painting

from options import Options
from tensorboard import TensorBoard
writer = None
local_it = 0 

def log_progress(painting):
    global local_it
    local_it +=1
    if local_it %5==0:
        with torch.no_grad():
            np_painting = painting(strokes=strokes_small).detach().cpu().numpy()[0].transpose(1,2,0)
            opt.writer.add_image('images/planasdfasdf', np.clip(np_painting, a_min=0, a_max=1), local_it)

def log_painting(painting, step, name='images/plan_all_strokes'):
    np_painting = painting(strokes=strokes_small).detach().cpu().numpy()[0].transpose(1,2,0)
    opt.writer.add_image(name, np.clip(np_painting, a_min=0, a_max=1), step)

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()


    b = './painting'
    all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
    tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
    if '_planner' not in tensorboard_dir:
        tensorboard_dir += '_planner'

    writer = TensorBoard(tensorboard_dir)
    opt.writer = writer


    painting = plan_all_strokes(opt)

    # Export the strokes
    f = open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), "w")
    f.write(painting.to_csv())
    f.close()
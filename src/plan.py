import pickle
import numpy as np
import torch
from torchvision import transforms
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from paint_utils import save_colors

from options import Options
from tensorboard import TensorBoard

# from torch_painting_models import *
from torch_painting_models_continuous import *
from style_loss import compute_style_loss
from sketch_loss.sketch_loss import compute_sketch_loss, compute_canny_loss


from clip_loss import clip_conv_loss, clip_model, clip_text_loss, clip_model_16, clip_fc_loss
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
    colors = (torch.from_numpy(np.array(colors)) / 255.).float().to(device)
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

writer = None
local_it = 0 

def log_progress(painting, force_log=False, title='plan'):
    global local_it
    local_it +=1
    if (local_it %5==0) or force_log:
        with torch.no_grad():
            np_painting = painting(h,w, use_alpha=False).detach().cpu().numpy()[0].transpose(1,2,0)
            opt.writer.add_image('images/{}'.format(title), np.clip(np_painting, a_min=0, a_max=1), local_it)


def sort_brush_strokes_by_color(painting, bin_size=3000):
    with torch.no_grad():
        brush_strokes = [bs for bs in painting.brush_strokes]
        for j in range(0,len(brush_strokes), bin_size):
            brush_strokes[j:j+bin_size] = sorted(brush_strokes[j:j+bin_size], 
                key=lambda x : x.color_transform.mean()+x.color_transform.prod(), 
                reverse=True)
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


def random_init_painting(background_img, n_strokes):
    gridded_brush_strokes = []
    xys = [(x,y) for x in torch.linspace(-.95,.95,int(n_strokes**0.5)) \
                 for y in torch.linspace(-.95,.95,int(n_strokes**0.5))]
    random.shuffle(xys)
    for x,y in xys:
        # Random brush stroke
        brush_stroke = BrushStroke(xt=x, yt=y)
        gridded_brush_strokes.append(brush_stroke)

    painting = Painting(0, background_img=background_img, 
        brush_strokes=gridded_brush_strokes).to(device)
    return painting

local_it = 0
def parse_objective(objective_type, objective_data, p, weight=1.0):
    ''' p is the rendered painting '''
    global local_it
    if objective_type == 'text':
        return clip_text_loss(p, objective_data, opt.num_augs)[0] * weight
    elif objective_type == 'style':
        return compute_style_loss(p, objective_data) * weight
    elif objective_type == 'clip_conv_loss':
        return clip_conv_loss(p, objective_data) * weight
    elif objective_type == 'l2':
        return ((p - objective_data)**2).mean() * weight
    elif objective_type == 'clip_fc_loss':
        return clip_fc_loss(p, objective_data, opt.num_augs)[0] * weight
    elif objective_type == 'sketch':
        local_it += 1
        return compute_sketch_loss(objective_data, p, writer=writer, it=local_it) * weight
        # return compute_sketch_loss(objective_data, p, comparator=clip_conv_loss) * weight
        def clip_sketch_comparison(sketch, p):
            return clip_conv_loss(K.color.grayscale_to_rgb(sketch), K.color.grayscale_to_rgb(p))
        return compute_sketch_loss(objective_data, p,
            comparator=clip_sketch_comparison, writer=writer, it=local_it) * weight
    elif objective_type == 'canny':
        local_it += 1
        return compute_canny_loss(objective_data, p, writer=writer, it=local_it) * weight
        # def clip_sketch_comparison(sketch, p):
        #     return clip_conv_loss(K.color.grayscale_to_rgb(sketch), K.color.grayscale_to_rgb(p))
        # return compute_canny_loss(objective_data, p,
        #     comparator=clip_sketch_comparison, writer=writer, it=local_it) * weight


    else:
        print('don\'t know what objective')
        1/0

def save_painting_strokes(painting, opt):
    # brush_stroke = BrushStroke(random.choice(strokes_small)).to(device)
    canvas = transforms.Resize(size=(h,w))(painting.background_img)

    individual_strokes = torch.empty((len(painting.brush_strokes),canvas.shape[1], canvas.shape[2], canvas.shape[3]))
    running_strokes = torch.empty((len(painting.brush_strokes),canvas.shape[1], canvas.shape[2], canvas.shape[3]))

    with torch.no_grad():
        for i in range(len(painting.brush_strokes)):                
            single_stroke = painting.brush_strokes[i](h,w)

            canvas = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
            
            running_strokes[i] = canvas 
            individual_strokes[i] = single_stroke

    running_strokes = (running_strokes.detach().cpu().numpy().transpose(0,2,3,1)*255).astype(np.uint8)
    individual_strokes = (individual_strokes.detach().cpu().numpy().transpose(0,2,3,1)*255).astype(np.uint8)

    with open(os.path.join(painter.opt.cache_dir, 'running_strokes.npy'), 'wb') as f:
        np.save(f, running_strokes)
    with open(os.path.join(painter.opt.cache_dir, 'individual_strokes.npy'), 'wb') as f:
        np.save(f, individual_strokes)

    for i in range(len(running_strokes)):
        if i % 5 == 0 or i == len(running_strokes)-1:
            opt.writer.add_image('images/plan', running_strokes[i], i)
    for i in range(len(individual_strokes)):
        if i % 5 == 0 or i == len(running_strokes)-1:
            opt.writer.add_image('images/individual_strokes', individual_strokes[i], i)


def create_tensorboard():
    def new_tb_entry():
        import datetime
        date_and_time = datetime.datetime.now()
        run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
        return 'painting/{}_planner'.format(run_name)
    try:
        try:
            import google.colab
            IN_COLAB = True
        except:
            IN_COLAB = False
        if IN_COLAB:
            tensorboard_dir = new_tb_entry()
        else:
            b = './painting'
            all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
            tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
            if '_planner' not in tensorboard_dir:
                tensorboard_dir += '_planner'
    except:
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
    return np.clip(np_painting, a_min=0, a_max=1)

def plan(opt):
    global colors
    painting = random_init_painting(current_canvas, opt.num_strokes)

    # Do initilization objective(s)
    painting.to(device)

    if opt.init_objective:
        optim = torch.optim.RMSprop(painting.parameters(), lr=opt.init_lr) # Coarse optimization
        for j in tqdm(range(opt.init_optim_iter), desc="Initializing"):
            optim.zero_grad()
            p = painting(h, w, use_alpha=False)
            loss = 0
            for k in range(len(opt.init_objective)):
                loss += parse_objective(opt.init_objective[k], 
                    init_objective_data[k], p[:,:3], weight=opt.init_objective_weight[k])

            loss.backward()
            optim.step()
            painting.validate()
            optim.param_groups[0]['lr'] = optim.param_groups[0]['lr'] * 0.95
            # painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            painting = randomize_brush_stroke_order(painting)
            log_progress(painting, title='init_optimization')#, force_log=True)
            

    # Intermediate optimization. Do it a few times, and pick the best
    init_painting = copy.deepcopy(painting.cpu())
    best_init_painting, best_init_loss = init_painting, 999999999
    painting.to(device)
    for attempt in range(opt.n_inits):
        painting = copy.deepcopy(init_painting).to(device)
        # optim = torch.optim.RMSprop(painting.parameters(), lr=5e-3) # Coarse optimization
        optims = painting.get_optimizers(multiplier=opt.lr_multiplier)
        for j in tqdm(range(opt.intermediate_optim_iter), desc="Intermediate Optimization"):
            #optim.zero_grad()
            for o in optims: o.zero_grad()
            p = painting(h, w, use_alpha=False)
            loss = 0
            for k in range(len(opt.objective)):
                loss += parse_objective(opt.objective[k], 
                    objective_data[k], p[:,:3], weight=opt.objective_weight[k])
            loss.backward()
            # optim.step()
            for o in optims: o.step()
            painting.validate()
            for o in optims: o.param_groups[0]['lr'] = o.param_groups[0]['lr'] * 0.95
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            # make sure hidden strokes get some attention
            # painting = randomize_brush_stroke_order(painting)
            log_progress(painting, title='int_optimization_{}'.format(attempt))#, force_log=True)
            
        if loss.item() < best_init_loss:
            best_init_loss = loss.item()
            best_init_painting = copy.deepcopy(painting.cpu())
            painting.to(device)
            print('best_painting {}'.format(attempt))
    painting = best_init_painting.to(device)

    # Create the plan
    position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt \
                = painting.get_optimizers(multiplier=opt.lr_multiplier)
    optims = (position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt)
    for i in tqdm(range(opt.optim_iter), desc='Optimizing {} Strokes'.format(str(len(painting.brush_strokes)))):
        for o in optims: o.zero_grad()

        p = painting(h, w, use_alpha=False)
        
        loss = 0
        for k in range(len(opt.objective)):
            loss += parse_objective(opt.objective[k], 
                objective_data[k], p[:,:3], weight=opt.objective_weight[k])
        loss.backward()

        position_opt.step()
        rotation_opt.step()
        bend_opt.step()
        length_opt.step()
        thickness_opt.step()
        if i < .8*opt.optim_iter: color_opt.step()

        # position_opt.param_groups[0]['lr'] = position_opt.param_groups[0]['lr'] * 0.99
        # rotation_opt.param_groups[0]['lr'] = rotation_opt.param_groups[0]['lr'] * 0.99
        # color_opt.param_groups[0]['lr'] = color_opt.param_groups[0]['lr'] * 0.99
        # bend_opt.param_groups[0]['lr'] = bend_opt.param_groups[0]['lr'] * 0.99
        # length_opt.param_groups[0]['lr'] = length_opt.param_groups[0]['lr'] * 0.99
        # thickness_opt.param_groups[0]['lr'] = thickness_opt.param_groups[0]['lr'] * 0.99

        painting.validate()

        painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
        
        if i < 0.3*opt.optim_iter and i %3 == 0:
            # make sure hidden strokes get some attention
            painting = randomize_brush_stroke_order(painting)

        if (i % 10 == 0 and i > (0.5*opt.optim_iter)) or i > 0.9*opt.optim_iter:
            if opt.use_colors_from is None:
                # Cluster the colors from the existing painting
                colors = painting.cluster_colors(opt.n_colors)

            discretize_colors(painting, colors)
        log_progress(painting)#, force_log=True)

    if opt.use_colors_from is None:
        colors = painting.cluster_colors(opt.n_colors)
    with open(os.path.join(opt.cache_dir, 'colors_updated.npy'), 'wb') as f:
        np.save(f, (colors.detach().cpu().numpy()*255).astype(np.uint8))


    discretize_colors(painting, colors)
    painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
    log_progress(painting, force_log=True)
    # save_painting_strokes(painting, opt)
    return painting

def adapt(opt):
    # Load available colors
    with open(os.path.join(opt.cache_dir, 'colors_updated.npy'), 'rb') as f:
        colors = (torch.from_numpy(np.array(np.load(f))) / 255.).to(device)

    # Load past plan
    with open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), 'r') as fp:
        instructions = [parse_csv_line_continuous(line) for line in fp.readlines()] 
    brush_strokes = []
    for instruction in instructions[int(opt.remove_prop*opt.strokes_before_adapting):]:
        x, y, r, length, thickness, bend, color = instruction
        brush_strokes.append(BrushStroke( 
            a=r, xt=x*2-1, yt=y*2-1, color=torch.from_numpy(color).float(),
            stroke_length=torch.ones(1)*length,
            stroke_z=torch.ones(1)*thickness,
            stroke_bend=torch.ones(1)*bend))

    painting = Painting(0, background_img=current_canvas, brush_strokes=brush_strokes).to(device)
    
    with torch.no_grad():
        p = painting(h,w, use_alpha=False)
        opt.writer.add_image('images/plan_update{}'.format(opt.global_it), format_img(p), 0)

    painting.validate()
    
    # Optimize all brush strokes
    position_opt, rotation_opt, color_opt, bend_opt, length_opt, thickness_opt \
            = painting.get_optimizers(multiplier=opt.lr_multiplier*.25)
    # print(opt.objective)
    for j in tqdm(range(opt.adapt_optim_iter), desc='Optimizing {} Strokes'.format(str(len(painting.brush_strokes)))):
        position_opt.zero_grad()
        rotation_opt.zero_grad()
        color_opt.zero_grad()
        bend_opt.zero_grad()
        length_opt.zero_grad()
        thickness_opt.zero_grad()

        p = painting(h,w, use_alpha=False)
        loss = 0
        for k in range(len(opt.objective)):
            loss += parse_objective(opt.objective[k], 
                objective_data[k], p[:,:3], weight=opt.objective_weight[k])
        loss.backward()

        position_opt.step()
        rotation_opt.step()
        if j < .8*opt.adapt_optim_iter: color_opt.step()
        bend_opt.step()
        length_opt.step()
        thickness_opt.step()

        position_opt.param_groups[0]['lr'] = position_opt.param_groups[0]['lr'] * 0.99
        rotation_opt.param_groups[0]['lr'] = rotation_opt.param_groups[0]['lr'] * 0.99
        color_opt.param_groups[0]['lr'] = color_opt.param_groups[0]['lr'] * 0.99
        bend_opt.param_groups[0]['lr'] = bend_opt.param_groups[0]['lr'] * 0.99
        length_opt.param_groups[0]['lr'] = length_opt.param_groups[0]['lr'] * 0.99
        thickness_opt.param_groups[0]['lr'] = thickness_opt.param_groups[0]['lr'] * 0.99

        painting.validate()

        if j%2 == 0 and j > 0.5*opt.adapt_optim_iter:
            painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
            discretize_colors(painting, colors)
        # log_progress(painting)#, force_log=True)

        if j%5 == 0 or j == opt.adapt_optim_iter-1:
            opt.writer.add_image('images/plan_update{}'.format(opt.global_it), format_img(p), j+1)

    painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)
    discretize_colors(painting, colors)
    painting = sort_brush_strokes_by_color(painting, bin_size=opt.bin_size)

    with torch.no_grad():
        p = painting(h,w, use_alpha=False)
        opt.writer.add_image('images/plan_update{}'.format(opt.global_it), format_img(p), opt.adapt_optim_iter+1)

    return painting 

def load_objectives_data(opt):
    # Load Initial objective data
    global init_objective_data
    init_objective_data = [] 
    for i in range(len(opt.init_objective) if opt.init_objective else 0):
        if opt.init_objective[i] == 'text':
            with torch.no_grad():
                text_features = clip_model.encode_text(clip.tokenize(opt.init_objective_data[i]).to(device))
                init_objective_data.append(text_features)
        else:
            # Must be an image
            img = load_img(opt.init_objective_data[i],h=h, w=w).to(device)/255.
            init_objective_data.append(img)
            opt.writer.add_image('target/init_input{}'.format(i), format_img(img), 0)

    # Load Objective data
    global objective_data
    objective_data = [] 
    for i in range(len(opt.objective)):
        if opt.objective[i] == 'text':
            with torch.no_grad():
                text_features = clip_model.encode_text(clip.tokenize(opt.objective_data[i]).to(device))
                objective_data.append(text_features)
        else:
            # Must be an image
            img = load_img(opt.objective_data[i],h=h, w=w).to(device)/255.
            objective_data.append(img)
            opt.writer.add_image('target/input{}'.format(i), format_img(img), 0)

if __name__ == '__main__':
    global opt
    opt = Options()
    opt.gather_options()

    # painting = calibrate(opt)

    opt.writer = create_tensorboard()

    global h, w, colors, current_canvas, text_features, style_img, sketch
    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    w = int((opt.max_height/h)*w)
    h = int(opt.max_height)
    # print('hw', h, w)

    colors = None
    if opt.use_colors_from is not None:
        colors = get_colors(cv2.resize(cv2.imread(opt.use_colors_from)[:,:,::-1], (256, 256)), 
                n_colors=opt.n_colors)
        opt.writer.add_image('paint_colors/using_colors_from_input', save_colors(colors), 0)

    # Get the background of painting to be the current canvas
    if os.path.exists(os.path.join(opt.cache_dir, 'current_canvas.jpg')):
        current_canvas = load_img(os.path.join(opt.cache_dir, 'current_canvas.jpg'), h=h, w=w).to(device)/255.
    else:
        current_canvas = torch.ones(1,3,h,w).to(device)
    load_objectives_data(opt)

    # Start Planning
    if opt.generate_whole_plan or not opt.adaptive:
        painting = plan(opt)
    else:
        painting = adapt(opt)

    # Export the strokes
    f = open(os.path.join(opt.cache_dir, "next_brush_strokes.csv"), "w")
    f.write(painting.to_csv())
    f.close()
import glob
import json
import numpy as np
import pickle
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode, Resize

from brush_stroke import BrushStroke
bicubic = InterpolationMode.BICUBIC
from torch import nn

import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import gzip
from torchvision.transforms.functional import affine

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import matplotlib
def show_img(img, display_actual_size=True, title=""):
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
    plt.title(title)
    # plt.scatter(img.shape[1]/2, img.shape[0]/2)
    plt.show()


class MatMulLayer(nn.Module):
    def __init__(self, size, layers, activation=torch.nn.LeakyReLU):
        super(MatMulLayer, self).__init__()
        self.activation = activation()
        self.layers = layers
        self.weights = torch.nn.Parameter(torch.ones((layers,)+size, dtype=torch.float))
        self.biases = torch.nn.Parameter(torch.zeros((layers,)+size, dtype=torch.float))
    def forward(self, x):
        for i in range(self.layers):
            x = x * self.weights[i] + self.biases[i]
            if i != (self.layers-1):
                x = self.activation(x)
        return x

class BezierRenderer(nn.Module):
    def __init__(self, size_x, size_y, num_pts):
        super(BezierRenderer, self).__init__()
        self.size_x = size_x # grid dimensions (size_x x size_y)
        self.size_y = size_y
        self.P = num_pts
        
        self.set_render_size(size_x, size_y)

        self.weights = nn.Parameter(torch.eye(2)) # multiply each point in trajectory by this
        self.biases = nn.Parameter(torch.zeros(2)) # add this to each point
        
        self.thick_mult = nn.Parameter(torch.ones((1))*0.5)
        self.thick_bias = nn.Parameter(torch.zeros((1))+0.1)
        self.dark_exp = nn.Parameter(torch.ones((1)))
        self.dark_non_lin_weight = nn.Parameter(torch.ones((1))*0.5)

        # self.nc = self.P
        # self.conv = nn.Sequential(
        #     nn.Conv2d(self.P, self.nc, kernel_size=5, padding='same', dilation=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(self.nc, self.P, kernel_size=5, padding='same', dilation=1),
        # )
        # self.conv = nn.Sequential(
        #     # nn.Sigmoid(),
        #     nn.Conv2d(self.P, self.P, kernel_size=5, padding='same', dilation=1)
        # )

        # self.conv = MatMulLayer(size=(1,self.P-1,size_y,size_x), layers=1)

    def set_render_size(self, size_x, size_y):
        '''
        Set the internal dimensions that strokes should be rendered at
        '''
        self.size_x = size_x 
        self.size_y = size_y
        idxs_x = torch.arange(size_x) / size_x
        idxs_y = torch.arange(size_y) / size_y
        x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # G x G
        self.grid_coords = torch.stack((x_coords, y_coords), dim=2).reshape(1,size_y, size_x,2) # 1 x G x G x 2

    def forward(self, trajectories, thicknesses, use_conv):        
        # trajectories: (n, 2, self.P)
        # thicknesses: (n, 1, self.P)
        strokes = []
        for i in range(len(trajectories)):
            # Expand num_ctrl_pts to self.P (smooths the curve)
            stroke = self.curve_to_stroke(trajectories[i]) # (P, 2) range:[0,1]

            thickness = (thicknesses[i]*2 + 0.5) / 64
            # Stroke trajectory to bitmap
            stroke = self.render_stroke(stroke, thickness, use_conv=use_conv)
            strokes.append(stroke)
        strokes = torch.stack(strokes, dim=0)
        return strokes

    def curve_to_stroke(self, curve):
        # curve: (2, self.P)
        return torch.matmul(self.weights, curve).T + self.biases # (self.P, 2)

    def render_stroke(self, stroke, t, use_conv):
        # stroke: (P, 2)
        # t: (1, P)
        n = len(stroke)
        vs = stroke[:-1].reshape((-1,1,1,2)) # (P-1, 1, 1, 2)
        vs = torch.tile(vs, (1, self.size_y, self.size_x, 1)) # (P-1, size_y, size_x, 2)

        ws = stroke[1:].reshape((-1,1,1,2)) # (P-1, 1, 1, 2)
        ws = torch.tile(ws, (1, self.size_y, self.size_x, 1)) # (P-1, size_y, size_x, 2)

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)).to(ws.device) # (P-1, size_y, size_x, 2)

        # For each of the P segments, compute distance from every point to the line as well as the fraction along the line
        distances, fraction = self.dist_line_segment(coords, vs, ws) # (P-1, size_y, size_x)
        
        # Compute the start and end thickness of each line segment
        start_thickness = t[0, :-1].unsqueeze(1).unsqueeze(2) # (P-1, 1, 1)
        end_thickness = t[0, 1:].unsqueeze(1).unsqueeze(2) # (P-1, 1, 1)
        
        # Convert the fractions into thickness values
        thickness = start_thickness * (1 - fraction) + end_thickness * fraction # (P-1, size_y, size_x)
        thickness = thickness * self.thick_mult + self.thick_bias
        thickness = torch.clamp(thickness, min=1e-8)

        # Compute darkness for each line segment
        darkness = torch.clamp((thickness - distances)/(thickness), min=0.0, max=1.0) # (P-1, size_y, size_x)
        darkness = (darkness+1e-4)**self.dark_exp # (P-1, size_y, size_x)

        # Push furth towards 0 or 1
        dark_non_lin = 1/(1+torch.exp(-1*(darkness*16-8))) 
        
        # Weight this value
        darkness = self.dark_non_lin_weight*dark_non_lin \
             + (1-self.dark_non_lin_weight)*darkness

        import torchgeometry
        import warnings
        from brush_stroke import rigid_body_transform, rigid_body_transforms
        
        use_conv = False
        if use_conv:
            # Compute the angle the stroke segment is tilted at
            angles = torch.atan2((stroke[1:,0]-stroke[:-1,0]), (stroke[1:,1] - stroke[:-1,1]))

            # Create a transformation matrix to rotate and translate the stroke s.t. it is at middle
            # of the image and has no rotation
            M = rigid_body_transforms(a=angles,
                                    xt=(self.size_x/2-stroke[:-1,1]), # TODO: is it stroke[:-1] or stroke[1:] ? 
                                    yt=(self.size_y/2-stroke[:-1,0]), 
                                    anchor_x=stroke[:-1,1], 
                                    anchor_y=stroke[:-1,0])

            # Rotate and translate strokes
            with warnings.catch_warnings(): # suppress annoying torchgeometry warning
                warnings.simplefilter("ignore")
                darkness = torchgeometry.warp_perspective(darkness[:,None,:,:], 
                        M, dsize=(self.size_y,self.size_x)) # (P-1, 1, size_y, size_x)

            darkness = darkness[:,0,:,:].unsqueeze(0) # (1, P-1, size_y, size_x)


            darkness = self.conv(darkness) # (1, P-1, size_y, size_x)

            darkness = darkness.swapaxes(0,1) # (P-1, 1, size_y, size_x)

            # Undo the translation/rotaiton
            with warnings.catch_warnings(): # suppress annoing torchgeometry warning
                warnings.simplefilter("ignore")
                darkness = torchgeometry.warp_perspective(darkness, 
                        torch.inverse(M), dsize=(self.size_y,self.size_x))

            darkness = darkness.squeeze(1) # (P, size_y, size_x)

            # show_img(torch.max(darknesses, dim=0).values * dryness)

        # Max across channels to get final stroke
        darkness = torch.max(darkness, dim=0).values # (size_y, size_x)
        
        return darkness

    
    # distance from point p to line segment v--w
    # also returns the fraction of the length of v--w that the point p is along
    def dist_line_segment(self, p, v, w):
        d = torch.linalg.norm(v-w, dim=3) # (n-1) x G x G
        dot = (p-v) * (w-v)
        dot_sum = torch.sum(dot, dim=3) / (d**2 + 1e-5)
        t = dot_sum.unsqueeze(3) # (n-1) x G x G x 1
        t = torch.clamp(t, min=0, max=1) # (n-1) x G x G x 1
        proj = v + t * (w-v) # (n-1) x G x G x 2
        return torch.linalg.norm(p-proj, dim=3), t.squeeze(3)

    def validate(self):
        # Ensure that parameters are within some valid range
        #self.thick_exp.data.clamp_(min=1.0)
        pass

def get_param2img(opt, device='cuda'):
    param2img = StrokeParametersToImage()
    saved_model_fn = os.path.join(opt.cache_dir, 'param2img.pt')
    if os.path.exists(saved_model_fn):
        param2img.load_state_dict(torch.load(saved_model_fn))
    else:
        print('Instatiating param2img without a saved trained model. Just using defaults.')
    param2img.eval()
    param2img.requires_grad_(False)
    param2img.to(device)
    return param2img


def special_sigmoid(x):
    return 1/(1+torch.exp(-1.*((x*2-1)+0.2) / 0.05))
    # x[x < 0.1] = 1/(1+torch.exp(-1.*((x[x < 0.1]*2-1)+0.2) / 0.05))
    # return x

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn 
    return pp

def process_img(img):
    return np.clip(img.detach().cpu().numpy(), a_min=0, a_max=1)*255


class StrokeParametersToImage(nn.Module):
    def __init__(self, output_dim_meters=None):
        super(StrokeParametersToImage, self).__init__()
        self.size_x = 200#128#256
        self.size_y = 200#128#256

        if output_dim_meters is not None:
            # input parameters are in meters. Translate these to proportions for the renderer.
            # Also account for the starting position of the stroke (Which should be 0,0)
            self.x_m = 1.0#1.0 / output_dim_meters[1]
            self.y_m = 1.0#-1.0 / output_dim_meters[0]

            self.x_m = nn.Parameter(torch.tensor(self.x_m))
            self.y_m = nn.Parameter(torch.tensor(self.y_m))
        else:
            self.x_m = nn.Parameter(torch.tensor(1.0))
            self.y_m = nn.Parameter(torch.tensor(1.0))

        self.x_b = nn.Parameter(torch.tensor(0.0))
        self.y_b = nn.Parameter(torch.tensor(0.0))
        
        self.renderer = BezierRenderer(size_x=self.size_x, 
                                       size_y=self.size_y,
                                       num_pts=32)

    def get_rotated_trajectory(self, rotation, trajectory):
        '''
        Rotate a trajectory some specified rotation (in radian).
        Rotates around (0,0) 
        args:
            rotation : torch.Tensor([radians])
            trajectory : torch.Tensor((1,n_points,2)) y=dim_0
        '''
        x, y = trajectory[0,:,1], trajectory[0,:,0]

        x_rot = x*torch.cos(rotation) + y*torch.sin(rotation)
        y_rot = -1*x*torch.sin(rotation) + y*torch.cos(rotation)

        return torch.stack([y_rot, x_rot], dim=1).unsqueeze(0)

    def forward(self, parameter, x_start, y_start, theta, use_conv):
        '''
        args:
            parameter : (batch, num_pts, 3)
            x_start : where to start stroke. range [0,1]
            y_start : where to start stroke. range [0,1]
            theta : how much to rotate stroke. Radians.
            use_conv : 
        '''
        # parameter: (batch, num_pts, 3)
        traj = parameter[:,:,0:2] # (batch, num_pts, 2)
        thicknesses = parameter[:,:,2:3] # (batch, num_pts, 1)

        # x,y to y,x
        traj = torch.flip(traj, dims=(2,)) 

        # To be consistent with the way BrushStroke.execute runs trajectories
        traj[:,:,0] = -1.0*traj[:,:,0] # y = -y yeah weird

        # Rotate before converting to proportions (errors when canvas isn't square)
        traj = self.get_rotated_trajectory(theta, traj)
        
        # traj from meters to proportion
        traj[:,:,0] *= self.y_m
        traj[:,:,1] *= self.x_m 

        traj[:,:,0] += self.y_b
        traj[:,:,1] += self.x_b

        # From traj starting  x=0,y=0 to  x=x_start, y=y_start
        traj[:,:,0] += y_start
        traj[:,:,1] += x_start

        # batch, ctrl_point, 2 -> batch, 2, ctrl_points
        traj = torch.permute(traj, (0,2,1))
        thicknesses = torch.permute(thicknesses, (0,2,1))
        
        # Run through Differentiable renderer
        x = self.renderer(traj,# * self.traj_factor[None] + self.traj_bias, 
                          thicknesses,
                          use_conv=use_conv) # (batch, size_y, size_x)
        
        return x
    
l1_loss = nn.L1Loss()
# def shift_invariant_loss(pred, real, before, fnl_weight=0.5, n=3, delta=0.02):
#     losses = None
#     for dx in torch.linspace(start=-1.0*delta, end=delta, steps=n):
#         for dy in torch.linspace(start=-1.0*delta, end=delta, steps=n):
#             x = int(dx*real.shape[2])
#             y = int(dy*real.shape[1])
#             # Translate the real stroke slightly
#             translated_pred  = affine(pred, angle=0, translate=(x, y), fill=0, scale=1.0, shear=0)

#             # L2
#             diff = (translated_pred - real)**2
#             l = diff.mean(dim=(1,2))
            
#             losses = l[None,:] if losses is None else torch.cat([losses, l[None,:]], dim=0)

#     # Only use the loss from the shift that gave the least loss value
#     loss, inds = torch.min(losses, dim=0)

#     # Extra loss for missing a stroke
#     stroke_areas = (real - before).abs() > 0.1
#     false_negative_loss = l1_loss(pred[stroke_areas], real[stroke_areas]) * fnl_weight
    
#     return loss.mean() + l1_loss(pred, real) + false_negative_loss

def loss_fcn(pred, real, before, fnl_weight=0.5):
    # Extra loss for missing a stroke
    stroke_areas = (real - before).abs() > 0.1
    if fnl_weight > 0:
        false_negative_loss = l1_loss(pred[stroke_areas], real[stroke_areas]) * fnl_weight
    else:
        false_negative_loss = 0
    return l1_loss(pred, real) + false_negative_loss

def train_param2stroke(opt, device='cuda', n_log=8, batch_size=32):
    torch.random.manual_seed(0)
    np.random.seed(0)


    # Get lists of saved data files
    canvases_before_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'canvases_before_*.npy'))
    canvases_after_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'canvases_after_*.npy'))
    brush_strokes_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'stroke_parameters*.npy'))
    
    # Ensure each of the files is in the order they were recorded in
    canvases_before_fns = sorted(canvases_before_fns)
    canvases_after_fns = sorted(canvases_after_fns)
    brush_strokes_fns = sorted(brush_strokes_fns)

    # canvases_before_fns = canvases_before_fns[:5]
    # canvases_after_fns = canvases_after_fns[:5]
    # brush_strokes_fns = brush_strokes_fns[:5]

    # Load data
    canvases_before = None
    for canvases_before_fn in canvases_before_fns:
        with gzip.GzipFile(canvases_before_fn,'r') as f:
            s = np.load(f, allow_pickle=True).astype(np.float32)/255.
            canvases_before = s if canvases_before is None else np.concatenate([canvases_before, s])

    canvases_after = None
    for canvases_after_fn in canvases_after_fns:
        with gzip.GzipFile(canvases_after_fn,'r') as f:
            s = np.load(f, allow_pickle=True).astype(np.float32)/255.
            canvases_after = s if canvases_after is None else np.concatenate([canvases_after, s])

    brush_strokes = []
    for brush_strokes_fn in brush_strokes_fns:
        bs = pickle.load(open(brush_strokes_fn,'rb'))
        brush_strokes = bs if brush_strokes is None else np.concatenate([brush_strokes, bs]) 
    
    canvases_before = torch.from_numpy(canvases_before).float().nan_to_num()
    canvases_after = torch.from_numpy(canvases_after).float().nan_to_num()
    

    # Size to render for training
    render_height_pix = 175
    w_h_ratio = canvases_before.shape[2] / canvases_before.shape[1]
    h_render_pix, w_render_pix = render_height_pix, int(render_height_pix*w_h_ratio)

    # Size to render for logging to tensorboard
    h_log, w_log = h_render_pix, w_render_pix
    res_log = Resize((h_log, w_log), bicubic, antialias=True)

    n = len(canvases_before)

    # Randomize
    rand_ind = torch.randperm(n)
    # rand_ind = torch.argsort(parameters[:,1,3], descending=True) # Sort by length

    canvases_before = canvases_before[rand_ind]
    canvases_after = canvases_after[rand_ind]
    brush_strokes = brush_strokes[rand_ind]

    # Channels first (and invert for some reason)
    canvases_before = 1-torch.permute(canvases_before, (0,3,1,2))
    canvases_after = 1-torch.permute(canvases_after, (0,3,1,2))

    # Resize strokes the size they'll be predicted at
    canvases_before = transforms.Resize((h_render_pix,w_render_pix), bicubic, antialias=True)(canvases_before)
    canvases_after = transforms.Resize((h_render_pix,w_render_pix), bicubic, antialias=True)(canvases_after)
    
    canvases_before = canvases_before.to(device)
    canvases_after = canvases_after.to(device)

    # canvases_before.requires_grad = False 
    # canvases_after.requires_grad = False 
    for bs in brush_strokes:
        bs.color_transform = nn.Parameter(torch.zeros(3))
        bs.requires_grad_(False)

    # Train/Validation split
    val_prop = 0.25

    train_canvases_before = canvases_before[int(val_prop*n):]
    train_canvases_after = canvases_after[int(val_prop*n):]
    train_brush_strokes = brush_strokes[int(val_prop*n):]

    val_canvases_before = canvases_before[:int(val_prop*n)]
    val_canvases_after = canvases_after[:int(val_prop*n)]
    val_brush_strokes = brush_strokes[:int(val_prop*n)]
    print('{} training strokes. {} validation strokes'.format(len(train_canvases_before), len(val_canvases_before)))


    trans = StrokeParametersToImage(output_dim_meters=(opt.STROKE_LIBRARY_CANVAS_HEIGHT_M, opt.STROKE_LIBRARY_CANVAS_WIDTH_M))
    trans = trans.to(device)
    
    # Need to render at the correct ratio
    # trans.renderer.set_render_size(size_x=w_render_pix, size_y=h_render_pix)
    # trans.renderer.set_render_size(size_x=h_render_pix, size_y=h_render_pix)
    
    print('# parameters in Param2Image model:', get_n_params(trans))
    optim = torch.optim.Adam(trans.parameters(), lr=1e-3)


    # Keep track of the best model (judged by validation error)
    best_model = copy.deepcopy(trans)
    best_val_loss = 99999
    best_hasnt_changed_for = 0

    # Main Training Loop
    for it in tqdm(range(3000)):
        # When to start training conv
        train_conv = it > 300 

        # with torch.autograd.set_detect_anomaly(True):
        if best_hasnt_changed_for >= 400 and it > 1000:
            break # all done :)
        optim.zero_grad()

        loss = 0
        ep_loss = 0
        for batch_it in range(len(train_brush_strokes)):
            bs = train_brush_strokes[batch_it].to(device)
            canvas_before = train_canvases_before[batch_it:batch_it+1]
            canvas_after = train_canvases_after[batch_it:batch_it+1]

            isolated_stroke = bs(h=h_render_pix, w=w_render_pix, param2img=trans, use_conv=train_conv)

            # Pad 1 or two to make it fit
            if isolated_stroke.shape[2] != h_render_pix or isolated_stroke.shape[3] != w_render_pix:
                isolated_stroke = transforms.Resize((h_render_pix, w_render_pix), bicubic, antialias=True)(isolated_stroke)

            predicted_canvas = canvas_before[:,:3] * (1 - isolated_stroke[:,3:]) \
                + isolated_stroke[:,3:] * isolated_stroke[:,:3]
            

            loss += loss_fcn(predicted_canvas, canvas_after, canvas_before,
                                         fnl_weight=max(0, 0.5 * ((200-it)/200)))
            ep_loss += loss.item()

            if (batch_it+1) % batch_size == 0 or batch_it == len(train_brush_strokes)-1:
                if not torch.isnan(loss):
                    # Don't try backprop further than full_param
                    loss.backward(inputs=tuple(trans.parameters()), retain_graph=True)

                    torch.nn.utils.clip_grad_norm_(trans.parameters(), max_norm=1.0)
                    # if train_conv:
                    #     conv_param_names = []
                    #     for name, param in trans.renderer.conv.named_parameters():
                    #         conv_param_names.append(name)
                    #     for name, param in trans.named_parameters():
                    #         name = name.replace('renderer.conv.', '')
                    #         if name not in conv_param_names:
                    #             if param.grad is not None:
                    #                 param.grad *= 0

                    optim.step()
                else:
                    print('Nan')
                    break
                loss = 0
        if it%20 == 0:
            opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss/len(train_brush_strokes), it)
            
            for name, param in trans.named_parameters():
                try:
                    val = float(param)
                    opt.writer.add_scalar('parameter/{}'.format(name), val, it)
                except:
                    pass

        # Log images
        if it%50 == 0:
            trans.eval()
            # print('dry opac tings', trans.dry_opac_factor, trans.dry_opac_bias)
            # print(trans.renderer.thick_exp)
            # print(trans.thickness_factor.detach().cpu().numpy(), trans.renderer.thicc_fact.detach().cpu().numpy(), sep='\n')
            
            for log_bs, log_canvases_before, log_canvases_after, log_name in [
                        [train_brush_strokes, train_canvases_before, train_canvases_after, 'train'],
                        [val_brush_strokes, val_canvases_before, val_canvases_after, 'val'],
                    ]:
                whole_img = None
                for i_log in range(min(len(log_bs), n_log)):
                    bs = log_bs[i_log].to(device)
                    canvas_before = log_canvases_before[i_log:i_log+1]
                    canvas_after = log_canvases_after[i_log:i_log+1]

                    # Coordinates to crop to just the changed area of the canvas
                    changed_pix = torch.abs(canvas_before-canvas_after)[0].max(dim=0).values > 0.3
                    changed_pix_vert = torch.max(changed_pix, dim=0)[0]
                    changed_pix_hor = torch.max(changed_pix, dim=1)[0]
                    
                    ws, we = 0, len(changed_pix_vert)
                    for i in range(len(changed_pix_vert)):
                        if changed_pix_vert[i] > 0:
                            ws = i 
                            break
                    for i in range(len(changed_pix_vert)-1, 0, -1):
                        if changed_pix_vert[i] > 0:
                            we = i 
                            break
                    hs, he = 0, len(changed_pix_hor)
                    for i in range(len(changed_pix_hor)):
                        if changed_pix_hor[i] > 0:
                            hs = i 
                            break
                    for i in range(len(changed_pix_hor)-1, 0, -1):
                        if changed_pix_hor[i] > 0:
                            he = i 
                            break
                    p = int(0.07*h_log)
                    ws, we, hs, he = max(0, ws-p), min(w_log, we+p), max(0, hs-p), min(h_log, he+p)
                    if we <= ws: 
                        ws, we = 0, w_log
                    if he <= hs: 
                        hs, he = 0, h_log 
                    # print(hs, he, ws, we)

                    with torch.no_grad():
                        isolated_stroke = bs(h=h_render_pix, w=w_render_pix, param2img=trans, use_conv=train_conv)
                        predicted_canvas = canvas_before[:,:3] * (1 - isolated_stroke[:,3:]) \
                            + isolated_stroke[:,3:] * isolated_stroke[:,:3]
                        
                    row_img = torch.cat([
                                res_log(canvas_before), 
                                res_log(canvas_after), 
                                res_log(predicted_canvas), 
                                res_log(canvas_after[:,:,hs:he,ws:we]), 
                                res_log(predicted_canvas[:,:,hs:he,ws:we]),
                                ], dim=3)
                    whole_img = row_img if whole_img is None else torch.cat([whole_img, row_img], dim=2)
                whole_img = torch.permute(whole_img, (0,2,3,1))[0]
                # show_img(whole_img)
                opt.writer.add_image('images_stroke_modeling_stroke/'+log_name, 
                    process_img(whole_img).astype(np.uint8), it)
            trans.train()
        # Compute validation error. Save best model
        if it % 10 == 0:
            with torch.no_grad():
                trans.eval()
                val_loss, stroke_area_loss = 0, 0
                stroke_areas = torch.abs(val_canvases_after - val_canvases_before) > 0.1
                for i_val in range(len(val_brush_strokes)):
                    bs = val_brush_strokes[i_val].to(device)
                    canvas_before = val_canvases_before[i_val:i_val+1]
                    canvas_after = val_canvases_after[i_val:i_val+1]

                    isolated_stroke = bs(h=h_render_pix, w=w_render_pix, param2img=trans, use_conv=train_conv)
                    predicted_canvas = canvas_before[:,:3] * (1 - isolated_stroke[:,3:]) \
                        + isolated_stroke[:,3:] * isolated_stroke[:,:3]
                    val_loss += l1_loss(predicted_canvas, canvas_after).item()
                    stroke_area_loss += l1_loss(predicted_canvas[stroke_areas[i_val:i_val+1]], canvas_after[stroke_areas[i_val:i_val+1]])
                if val_loss < best_val_loss and it > 50:
                    best_val_loss = val_loss
                    best_hasnt_changed_for = 0
                    best_model = copy.deepcopy(trans)
                best_hasnt_changed_for += 10
                trans.train()
            opt.writer.add_scalar('loss/val_loss_stroke_model', val_loss/len(val_brush_strokes), it)
            opt.writer.add_scalar('loss/val_stroke_area_loss', stroke_area_loss/len(val_brush_strokes), it)

    torch.save(best_model.cpu().state_dict(), os.path.join(opt.cache_dir, 'param2img.pt'))

if __name__ == '__main__':
    from options import Options
    opt = Options()
    opt.gather_options()

    torch.manual_seed(0)
    np.random.seed(0)

    from paint_utils3 import create_tensorboard
    opt.writer = create_tensorboard()

    # Train brush strokes
    train_param2stroke(opt)
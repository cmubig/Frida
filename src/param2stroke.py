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

class BezierRenderer(nn.Module):
    def __init__(self, size_x, size_y, num_ctrl_pts):
        super(BezierRenderer, self).__init__()
        self.size_x = size_x # grid dimensions (size_x x size_y)
        self.size_y = size_y 
        self.P = 10 # number of pieces to split Bezier curve into
        idxs_x = torch.arange(size_x)
        idxs_y = torch.arange(size_y)
        x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # G x G
        self.grid_coords = torch.stack((x_coords, y_coords), dim=2).reshape(1,size_y, size_x,2).to(device) # 1 x G x G x 2

        self.num_ctrl_pts = num_ctrl_pts

        self.weights = torch.zeros((self.num_ctrl_pts, self.P)).to(device)
        gaus = torch.signal.windows.gaussian(self.P*2, std=2.0) * 0.75
        for i in range(self.num_ctrl_pts):
            start_ind = int(self.P - self.P*(i/(self.num_ctrl_pts-1)))
            self.weights[i,:] = gaus[start_ind:start_ind+self.P]
        self.weights = self.weights / torch.sum(self.weights, dim=0, keepdim=True)
        
        self.thicc_fact = torch.ones((self.P, self.num_ctrl_pts), dtype=torch.float).to(device)
        # self.thicc_fact = torch.ones((self.P), dtype=torch.float).to(device)


    def forward(self, trajectories, thicknesses):        
        # trajectories: (n, 2, num_ctrl_pts)
        # thicknesses: (n, 1, num_ctrl_pts)
        strokes = []
        for i in range(len(trajectories)):
            # Expand num_ctrl_pts to self.P (smooths the curve)
            stroke = self.curve_to_stroke(trajectories[i]) # (P+1, 2)
            stroke[:,0] *= self.size_y
            stroke[:,1] *= self.size_x

            thickness = thicknesses[i]*2 + 0.5
            # Stroke trajectory to bitmap
            stroke = self.render_stroke(stroke, thickness)
            strokes.append(stroke)
        strokes = torch.stack(strokes, dim=0)
        return strokes

    def curve_to_stroke(self, curve):
        # curve: (2, num_ctrl_pts)
        p1 = curve[:,0:1].T # (2, 1)
        p2 = curve[:,1:2].T # (2, 1)
        p3 = curve[:,2:3].T # (2, 1)
        p4 = curve[:,3:4].T # (2, 1)

        control_pts = torch.stack([p1, p2, p3, p4], dim=2) # (2, 1, num_ctrl_pts)
        sample_pts = torch.matmul(control_pts, self.weights.to(device)) # (2, 1, P)

        sample_pts = torch.permute(sample_pts, (0, 2, 1)) # (2, P, 1)
        sample_pts = torch.reshape(sample_pts, (-1, 2)) # (P, 2)

        sample_pts = torch.cat([sample_pts, curve[:,3:4].T]) # (P+1, 2)
        return sample_pts

    def render_stroke(self, stroke, t):
        # stroke: (P+1, 2)
        # t: (1, num_ctrl_points)
        n = len(stroke)
        vs = stroke[:-1].reshape((-1,1,1,2)) # (P, 1, 1, 2)
        vs = torch.tile(vs, (1, self.size_y, self.size_x, 1)) # (P, size_y, size_x, 2)

        ws = stroke[1:].reshape((-1,1,1,2)) # (P, 1, 1, 2)
        ws = torch.tile(ws, (1, self.size_y, self.size_x, 1)) # (P, size_y, size_x, 2)

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)) # (P, size_y, size_x, 2)

        # For each of the P segments, compute distance from every point to the line as well as the fraction along the line
        distances, fraction = self.dist_line_segment(coords, vs, ws) # (P, size_y, size_x)
        # darknesses = torch.clamp((2*t - distances)/(2*t), min=0.0, max=1.0) # (n-1) x G x G
        
        # Scale the thicknesses by learnable factor
        thick = 2 * self.thicc_fact @ t.T # (P, 1, 1)
        thick = thick[:,:,None] # (P, 1)
        thick = torch.clamp(thick, min=1e-5)

        # Scale the thicknesses by learnable factor (If you want thicc_fact to be (P,1))
        # thick = torch.nn.Upsample(size=(self.P), mode='linear')(t.unsqueeze(0)) # (1,1,P)
        # thick = thick[0,0] # (P,)
        # thick = thick * self.thicc_fact # (P,)
        # thick = thick[:,None, None] # (P, 1, 1)
        # thick = torch.clamp(thick, min=1e-5)

        # Incorporate the thickness params with the distance from line matrices to compute darknesses
        darknesses = torch.clamp((thick - distances)/(thick), 
                                 min=0.0, max=1.0) # (P, size_y, size_x)

        # Max across channels to get final stroke
        darknesses = torch.max(darknesses, dim=0).values # (size_y, size_x)
        
        return darknesses
    
    # distance from point p to line segment v--w
    # also returns the fraction of the length of v--w that the point p is along
    def dist_line_segment(self, p, v, w):
        d = torch.linalg.norm(v-w, dim=3) # (n-1) x G x G
        # print(d.shape)
        # plt.matshow(d.detach().cpu()[0])
        # plt.show()
        # plt.matshow(d.detach().cpu()[5])
        # plt.show()
        dot = (p-v) * (w-v)
        # plt.matshow(dot.detach().cpu()[0,:,:,0])
        # plt.show()
        dot_sum = torch.sum(dot, dim=3) / (d**2 + 1e-5)
        # print('dot sum', dot_sum.shape)
        t = dot_sum.unsqueeze(3) # (n-1) x G x G x 1
        t = torch.clamp(t, min=0, max=1) # (n-1) x G x G
        # print(v.shape, (t*(w-v)).shape)
        # plt.matshow((t*(w-v))[0].detach().cpu()[:,:,0])
        # plt.show()
        proj = v + t * (w-v) # (n-1) x G x G x 2
        # plt.matshow((proj)[0].detach().cpu()[:,:,0])
        # plt.show()
        return torch.linalg.norm(p-proj, dim=3), t

    def validate(self):
        # Ensure that parameters are within some valid range
        #self.thick_exp.data.clamp_(min=1.0)
        pass

def get_param2img(opt, param2img=None, device='cuda'):
    param2img = StrokeParametersToImage()
    param2img.load_state_dict(torch.load(
        os.path.join(opt.cache_dir, 'param2img.pt')))
    param2img.eval()
    param2img.requires_grad_(False)
    param2img.to(device)
    return param2img


def special_sigmoid(x):
    return 1/(1+torch.exp(-1.*((x*2-1)+0.2) / 0.05))
    # return x

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
    def __init__(self, output_dim_meters=None, margin_props=None):
        super(StrokeParametersToImage, self).__init__()
        self.size_x = 256
        self.size_y = 256

        if output_dim_meters is not None and margin_props is not None:
            # input parameters are in meters. Translate these to proportions for the renderer.
            # Also account for the starting position of the stroke
            self.x_m = 1.0 / output_dim_meters[1]
            self.x_b = margin_props[1]
            self.y_m = -1.0 / output_dim_meters[0]
            self.y_b = margin_props[0]

            self.x_m = nn.Parameter(torch.tensor(self.x_m))
            self.x_b = nn.Parameter(torch.tensor(self.x_b))
            self.y_m = nn.Parameter(torch.tensor(self.y_m))
            self.y_b = nn.Parameter(torch.tensor(self.y_b))
        else:
            self.x_m = nn.Parameter(torch.tensor(1.0))
            self.x_b = nn.Parameter(torch.tensor(0.0))
            self.y_m = nn.Parameter(torch.tensor(1.0))
            self.y_b = nn.Parameter(torch.tensor(0.0))

        
        # self.conv = nn.Sequential(
        #     nn.BatchNorm2d(14),
        #     nn.Conv2d(14, self.nc, kernel_size=5, padding='same', dilation=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.BatchNorm2d(self.nc),
        #     nn.Conv2d(self.nc, 1, kernel_size=5, padding='same', dilation=1),
        #     nn.Sigmoid()
        # )
        
        self.renderer = BezierRenderer(size_x=self.size_x, 
                                       size_y=self.size_y,
                                       num_ctrl_pts=4).to(device)

        self.thicc_fact = nn.Parameter(self.renderer.thicc_fact)
        self.renderer.thicc_fact = self.thicc_fact

        self.weights = nn.Parameter(self.renderer.weights)
        self.renderer.weights = self.weights

        self.thickness_factor = torch.ones([1], device=device)
        self.thickness_factor = nn.Parameter(self.thickness_factor)

        self.traj_factor = torch.ones([2,4], device=device)
        self.traj_factor = nn.Parameter(self.traj_factor)
        self.traj_bias = torch.zeros([2,4], device=device)
        self.traj_bias = nn.Parameter(self.traj_bias)

    def forward(self, parameter):
        traj = parameter[:,:,0:2]
        thicknesses = parameter[:,:,2:3]

        # x,y to y,x
        traj = torch.flip(traj, dims=(2,)) 

        # traj from meters to proportion
        traj[:,:,0] *= self.y_m
        traj[:,:,0] += self.y_b
        traj[:,:,1] *= self.x_m 
        traj[:,:,1] += self.x_b

        # batch, ctrl_point, 2 -> batch, 2, ctrl_points
        traj = torch.permute(traj, (0,2,1))
        thicknesses = torch.permute(thicknesses, (0,2,1))
        
        # Run through Differentiable renderer
        x = self.renderer(traj * self.traj_factor[None] + self.traj_bias, thicknesses * self.thickness_factor)
        x = x[:,None,:,:]

        # Below is if using a CNN to combine the distance matrices
        # thicknesses = thicknesses.squeeze()
        # # print(thicknesses.shape)
        # t = thicknesses[:,:,None,None].repeat(1,1,self.size_y, self.size_x)
        # # print(t.shape, x.shape)
        # x = torch.cat([x,t], dim=1)
        # x = self.conv(x)

        return x[:,0]
    


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
    false_negative_loss = l1_loss(pred[stroke_areas], real[stroke_areas]) * fnl_weight
    
    return l1_loss(pred, real) + false_negative_loss

def train_param2stroke(opt, device='cuda', n_log=6, batch_size=32):
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Size to render for training
    h_render_pix, w_render_pix = 256, 256
    # Size to render for logging to tensorboard
    h_log, w_log = 256, 256
    res_log = Resize((h_log, w_log), bicubic, antialias=True)

    # Get lists of saved data files
    canvases_before_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'canvases_before_*.npy'))
    canvases_after_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'canvases_after_*.npy'))
    brush_strokes_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'stroke_parameters*.npy'))
    
    # Ensure each of the files is in the order they were recorded in
    canvases_before_fns = sorted(canvases_before_fns)
    canvases_after_fns = sorted(canvases_after_fns)
    brush_strokes_fns = sorted(brush_strokes_fns)

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
    
    # Margins around the starting point of the stroke
    left_margin_prop = 0.1
    top_margin_prop = 0.5

    canvases_before = torch.from_numpy(canvases_before).float().nan_to_num()
    canvases_after = torch.from_numpy(canvases_after).float().nan_to_num()
    
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

    canvases_before.requires_grad = False 
    canvases_after.requires_grad = False 
    for bs in brush_strokes:
        bs.requires_grad = False
        bs.color_transform = nn.Parameter(torch.zeros(3)) # Assume black paint

    # Train/Validation split
    val_prop = 0.25

    train_canvases_before = canvases_before[int(val_prop*n):]
    train_canvases_after = canvases_after[int(val_prop*n):]
    train_brush_strokes = brush_strokes[int(val_prop*n):]

    val_canvases_before = canvases_before[:int(val_prop*n)]
    val_canvases_after = canvases_after[:int(val_prop*n)]
    val_brush_strokes = brush_strokes[:int(val_prop*n)]
    print('{} training strokes. {} validation strokes'.format(len(train_canvases_before), len(val_canvases_before)))


    trans = StrokeParametersToImage(output_dim_meters=(opt.STROKE_LIBRARY_CANVAS_HEIGHT_M, opt.STROKE_LIBRARY_CANVAS_WIDTH_M),
                                    margin_props=(top_margin_prop, left_margin_prop)) 
    trans.requires_grad_(True)
    trans = trans.to(device)
    print('# parameters in Param2Image model:', get_n_params(trans))
    optim = torch.optim.Adam(trans.parameters(), lr=1e-3)
    weights_optim = torch.optim.Adam([trans.renderer.weights], lr=1e-3)
    
    # Keep track of the best model (judged by validation error)
    best_model = copy.deepcopy(trans)
    best_val_loss = 99999
    best_hasnt_changed_for = 0

    # Do some pre-training on the weights of the renderer since its init. is bad
    from scipy.interpolate import make_interp_spline
    for it in tqdm(range(5000)):
        weights_optim.zero_grad()
        loss = 0
        for batch_it in range(batch_size):
            bs = BrushStroke(opt).to(device)
            
            # Get the robot's trajectory path
            path = bs.path.detach().cpu().numpy()
            t = range(0, len(path))

            b_x = make_interp_spline(t,path[:,0])
            b_y = make_interp_spline(t,path[:,1])
            steps = trans.renderer.P + 1
            t_new = np.linspace(0, len(path)-1, steps)
            x_new = b_x(t_new)
            y_new = b_y(t_new)
            true_path = torch.from_numpy(np.stack([x_new, y_new], axis=0).T).float().to(device)

            # Get path defined by renderer weights
            renderer_path = trans.renderer.curve_to_stroke(bs.path[:,:2].T)

            loss += l1_loss(renderer_path, true_path)
        loss.backward()
        weights_optim.step()
        if loss.item() < 0.001:
            break

    # Main Training Loop
    for it in tqdm(range(3000)):
        # with torch.autograd.set_detect_anomaly(True):
        if best_hasnt_changed_for >= 400 and it > 1200:
            break # all done :)
        optim.zero_grad()

        loss = 0
        ep_loss = 0
        for batch_it in range(len(train_brush_strokes)):
            bs = train_brush_strokes[batch_it].to(device)
            canvas_before = train_canvases_before[batch_it:batch_it+1]
            canvas_after = train_canvases_after[batch_it:batch_it+1]

            isolated_stroke = bs(h=h_render_pix, w=w_render_pix, param2img=trans)
            predicted_canvas = canvas_before[:,:3] * (1 - isolated_stroke[:,3:]) \
                + isolated_stroke[:,3:] * isolated_stroke[:,:3]

            loss += loss_fcn(predicted_canvas, canvas_after, canvas_before,
                                         fnl_weight=0.5 if it < 600 else 0.1)
            ep_loss += loss.item()

            if (batch_it+1) % batch_size == 0 or batch_it == len(train_brush_strokes)-1:
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trans.parameters(), max_norm=1.0)
                    if it < 1000:
                        # Optimize other parameters before weights (if optimized too soon it finds bad minima)
                        trans.renderer.weights.grad.data *= 0
                    else:
                        trans.renderer.weights.grad.data *= 0.1
                    optim.step()
                else:
                    print('Nan')
                    break
                loss = 0
        if it%10 == 0:
            opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss, it)

        # Log images
        if it%100 == 0:
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
                    changed_pix = torch.abs(canvas_before-canvas_after)[0].mean(dim=0) > 0.4
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
                        isolated_stroke = bs(h=h_render_pix, w=w_render_pix, param2img=trans)
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

                    isolated_stroke = bs(h=h_render_pix, w=w_render_pix, param2img=trans)
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
            opt.writer.add_scalar('loss/val_loss_stroke_model', val_loss, it)
            opt.writer.add_scalar('loss/val_stroke_area_loss', stroke_area_loss, it)

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
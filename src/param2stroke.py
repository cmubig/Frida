import glob
import json
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode 
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
        torch.set_printoptions(precision=3, sci_mode=False)
        print(self.weights)
        
        self.thicc_fact = torch.ones((self.P, self.num_ctrl_pts), dtype=torch.float).to(device)


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
        # print(distances.shape)
        # for i in range(len(distances)):
        #     plt.matshow(distances.detach().cpu()[i])
        #     plt.show()
        
        # Scale the thicknesses by learnable factor
        thick = 2 * self.thicc_fact @ t.T # (P, 1, 1)
        thick = thick[:,:,None] # (P, 1)
        
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

def get_param2img(opt, device='cuda'):
    w_canvas_m = opt.CANVAS_WIDTH_M
    h_canvas_m = opt.CANVAS_HEIGHT_M
    
    # Load how many meters the param2image model output represents
    with open(os.path.join(opt.cache_dir, 'param2stroke_settings.json'), 'r') as f:
        settings = json.load(f)
        print('Param2Stroke Settings:', settings)
        w_p2i_m = settings['w_p2i_m']
        h_p2i_m = settings['h_p2i_m']
        xtra_room_horz_m = settings['xtra_room_horz_m']
        xtra_room_vert_m = settings['xtra_room_vert_m']
        MAX_BEND = settings['MAX_BEND']
    
    # print(w_p2i_m - xtra_room_horz_m, 0.5*w_canvas_m)
    if (w_p2i_m- xtra_room_horz_m) > (0.5 * w_canvas_m):
        raise Exception("The canvas width is less than two times the max_stroke_length. This makes it really too hard to render. Must use larger canvas.")
    
    param2img = StrokeParametersToImage()
    param2img.load_state_dict(torch.load(
        os.path.join(opt.cache_dir, 'param2img.pt')))
    param2img.eval()
    param2img.to(device)

    def forward(param, h_render_pix, w_render_pix):
        # Figure out what to resize the output of param2image should be based on the desired render size
        w_p2i_render_pix = int((w_p2i_m / w_canvas_m) * w_render_pix)
        h_p2i_render_pix = int((h_p2i_m / h_canvas_m) * h_render_pix)
        res_to_render = transforms.Resize((h_p2i_render_pix, w_p2i_render_pix), bicubic)

        # Pad the output of param2image such that the start of the stroke is directly in the
        # middle of the canvas and the dimensions of the image match the render size
        pad_left_m = 0.5 * w_canvas_m - xtra_room_horz_m
        pad_right_m = w_canvas_m - pad_left_m - w_p2i_m
        pad_top_m = 0.5 * h_canvas_m - MAX_BEND - xtra_room_vert_m
        pad_bottom_m = 0.5 * h_canvas_m - MAX_BEND - xtra_room_vert_m

        pad_left_pix =   int(pad_left_m   * (w_render_pix / w_canvas_m))
        pad_right_pix =  int(pad_right_m  * (w_render_pix / w_canvas_m))
        pad_top_pix =    int(pad_top_m    * (h_render_pix / h_canvas_m))
        pad_bottom_pix = int(pad_bottom_m * (h_render_pix / h_canvas_m))

        pad_for_full = transforms.Pad((pad_left_pix, pad_top_pix, pad_right_pix, pad_bottom_pix))

        return pad_for_full(res_to_render(param2img(param)))
    return forward


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

def remove_background_noise(strokes):
    # Clear regions that aren't likely strokes. 
    # i.e. remove the non-stroke data, perceptual issues.
    # print('mean', strokes.mean())
    from scipy import ndimage
    stroke_mean = strokes.mean(dim=0)
    # plt.imshow(stroke_mean.cpu().numpy())
    # plt.colorbar()
    # plt.show()

    stroke_mean = ndimage.maximum_filter(stroke_mean, size=30)
    stroke_mean = torch.from_numpy(stroke_mean)
    
    # plt.imshow(stroke_mean)
    # plt.colorbar()
    # plt.show()
    # print(torch.quantile(stroke_mean, 0.1))
    unlikely_areas = (stroke_mean < torch.quantile(stroke_mean[stroke_mean > 0.001], 0.5))#[None,:,:]
    # plt.imshow(unlikely_areas*0.5 + strokes.mean(dim=0))
    # plt.colorbar()
    # plt.show()
    unlikely_areas = ndimage.minimum_filter(unlikely_areas, size=50) # Exapnd the region slightly
    unlikely_areas = torch.from_numpy(unlikely_areas)

    # plt.imshow(unlikely_areas)
    # plt.colorbar()
    # plt.show()

    # plt.imshow(unlikely_areas*0.5 + strokes.mean(dim=0))
    # plt.colorbar()
    # plt.show()
    strokes[:,unlikely_areas] = 0
    # print('mean', strokes.mean())
    return strokes

def log_images(imgs, labels, label, writer, step=0):
    fig, ax = plt.subplots(1, len(imgs), figsize=(5*len(imgs),5))

    for i in range(len(imgs)):
        # print(imgs[i].min(), imgs[i].max())
        ax[i].imshow(imgs[i], cmap='gray', vmin=0, vmax=255)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(labels[i])
    fig.tight_layout()
    writer.add_figure(label, fig, step)

class StrokeParametersToImage(nn.Module):
    def __init__(self, output_dim_meters=None, margin_props=None):
        super(StrokeParametersToImage, self).__init__()
        self.nc = 20
        self.size_x = 128
        self.size_y = 64

        if output_dim_meters is not None and margin_props is not None:
            # input parameters are in meters. Translate these to proportions for the renderer.
            # Also account for the starting position of the stroke
            self.x_m = 1 / output_dim_meters[1]
            self.x_b = margin_props[1]
            self.y_m = -1 / output_dim_meters[0]
            self.y_b = margin_props[0]

            self.x_m = nn.Parameter(torch.tensor(self.x_m))
            self.x_b = nn.Parameter(torch.tensor(self.x_b))
            self.y_m = nn.Parameter(torch.tensor(self.y_m))
            self.y_b = nn.Parameter(torch.tensor(self.y_b))
        else:

            self.x_m = nn.Parameter(torch.tensor(0.0))
            self.x_b = nn.Parameter(torch.tensor(0.0))
            self.y_m = nn.Parameter(torch.tensor(0.0))
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

        self.renderer.thicc_fact.requires_grad = True
        self.thicc_fact = nn.Parameter(self.renderer.thicc_fact)
        self.renderer.thicc_fact = self.thicc_fact

        self.renderer.weights.requires_grad = True
        self.weights = nn.Parameter(self.renderer.weights)
        self.renderer.weights = self.weights

        self.thickness_factor = torch.ones([1], device=device, requires_grad=True)
        self.thickness_factor = nn.Parameter(self.thickness_factor)

        self.traj_factor = torch.ones([2,4], device=device, requires_grad=True)
        self.traj_factor = nn.Parameter(self.traj_factor)
        self.traj_bias = torch.zeros([2,4], device=device, requires_grad=True)
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
def shift_invariant_loss(pred, real, n=8, delta=0.02):
    losses = None
    for dx in torch.linspace(start=-1.0*delta, end=delta, steps=n):
        for dy in torch.linspace(start=-1.0*delta, end=delta, steps=n):
            x = int(dx*real.shape[2])
            y = int(dy*real.shape[1])
            # Translate the real stroke slightly
            translated_pred  = affine(pred, angle=0, translate=(x, y), fill=0, scale=1.0, shear=0)

            # L2
            diff = (translated_pred - real)**2
            l = diff.mean(dim=(1,2))
            
            losses = l[None,:] if losses is None else torch.cat([losses, l[None,:]], dim=0)

    # Only use the loss from the shift that gave the least loss value
    loss, inds = torch.min(losses, dim=0)
    # from scipy import stats
    # mode = stats.mode(inds.cpu().numpy())
    # print(mode, mode.mode, mode.count)
    return loss.mean() + l1_loss(pred, real)

def train_param2stroke(opt, device='cuda'):
    # param2img = get_param2img(opt)
    # x = param2img(torch.zeros(1,4, device=device), 200, 400)
    # print(x.shape)
    # plt.imshow(x[0].cpu().detach().numpy())
    # plt.show()
    torch.random.manual_seed(0)
    np.random.seed(0)

    # Load data
    stroke_img_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'stroke_intensities*.npy'))
    stroke_param_fns = glob.glob(os.path.join(opt.cache_dir, 'stroke_library', 'stroke_parameters*.npy'))
    stroke_img_fns = sorted(stroke_img_fns)
    stroke_param_fns = sorted(stroke_param_fns)
    # print(stroke_img_fns, stroke_param_fns)

    strokes = None
    for stroke_img_fn in stroke_img_fns:
        with gzip.GzipFile(stroke_img_fn,'r') as f:
            s = np.load(f, allow_pickle=True).astype(np.float32)/255.

            h_og, w_og = s[0].shape[0], s[0].shape[1]
            # Crop so that we only predict around the area around the stroke not all the background blank pix
            xtra_room_horz_m = 0.015 # Added padding to ensure we don't crop out actual paint
            xtra_room_vert_m = 0.005
            # Do in meters first. ws_m = width start crop in meters
            ws_m = (opt.STROKE_LIBRARY_CANVAS_WIDTH_M/2) - xtra_room_horz_m
            we_m = ws_m + xtra_room_horz_m*2 + opt.MAX_STROKE_LENGTH
            hs_m = (opt.STROKE_LIBRARY_CANVAS_HEIGHT_M/2) - opt.MAX_BEND - xtra_room_vert_m
            he_m = hs_m + 2*xtra_room_vert_m + 2*opt.MAX_BEND
            # Convert from meters to pix
            pix_per_m = w_og / opt.STROKE_LIBRARY_CANVAS_WIDTH_M
            ws, we, hs, he = ws_m*pix_per_m, we_m*pix_per_m, hs_m*pix_per_m, he_m*pix_per_m
            ws, we, hs, he = int(ws), int(we), int(hs), int(he)
            # print(ws, we, hs, he)
            s = s[:, hs:he, ws:we]

            strokes = s if strokes is None else np.concatenate([strokes, s])

    parameters = None
    for stroke_param_fn in stroke_param_fns:
        p = np.load(stroke_param_fn, allow_pickle=True, encoding='bytes') 
        parameters = p if parameters is None else np.concatenate([parameters, p]) 
    
    render_size_width_m = xtra_room_horz_m*2 + opt.MAX_STROKE_LENGTH
    render_size_height_m = 2*xtra_room_vert_m + 2*opt.MAX_BEND
    left_margin_prop = xtra_room_horz_m / render_size_width_m
    top_margin_prop = 0.5

    strokes = torch.from_numpy(strokes).float().nan_to_num()
    parameters = torch.from_numpy(parameters.astype(np.float32)).float().nan_to_num()
    
    n = len(strokes)

    # Randomize
    rand_ind = torch.randperm(strokes.shape[0])
    # rand_ind = torch.argsort(parameters[:,1,3], descending=True) # Sort by length
    strokes = strokes[rand_ind]
    parameters = parameters[rand_ind]

    # Discrete. Makes the model push towards making bolder strokes
    strokes[strokes >= 0.5] = 1.
    strokes[strokes < 0.35] = 0. # Make sure background is very much gone

    # Save the amount of meters that the output of the param2image model represents
    w_p2i_m = we_m - ws_m 
    h_p2i_m = he_m - hs_m 
    with open(os.path.join(opt.cache_dir, 'param2stroke_settings.json'), 'w') as f:
        settings = {}
        settings['w_p2i_m'] = w_p2i_m
        settings['h_p2i_m'] = h_p2i_m
        settings['xtra_room_horz_m'] = xtra_room_horz_m
        settings['xtra_room_vert_m'] = xtra_room_vert_m
        settings['MAX_BEND'] = opt.MAX_BEND
        json.dump(settings, f, indent=4)


    # strokes = remove_background_noise(strokes)

    # Resize strokes the size they'll be predicted at
    t = StrokeParametersToImage()
    strokes = transforms.Resize((t.size_y,t.size_x), bicubic)(strokes)

    # for i in range(len(strokes)):
    #     strokes[i] -= strokes[i].min()
    #     if strokes[i].max() > 0.01:
    #         strokes[i] /= strokes[i].max()
    #     # strokes[i] *= 0.95
    #     # print(strokes[i].min(), strokes[i].max())
    
    # Filter out strokes that are bad perception. Avg is too high.
    # One bad apple can really spoil the bunch
    good_strokes = []
    good_parameters = []
    for i in range(len(strokes)):
        if strokes[i].mean() < 0.4: 
            good_strokes.append(strokes[i])
            good_parameters.append(parameters[i])
    print(len(strokes)- len(good_strokes), 'strokes removed because average value too high')
    strokes = torch.stack(good_strokes, dim=0)
    parameters = torch.stack(good_parameters, dim=0)

    strokes = strokes.to(device)
    parameters = parameters.to(device)


    trans = StrokeParametersToImage(output_dim_meters=(render_size_height_m, render_size_width_m),
                                    margin_props=(top_margin_prop, left_margin_prop)) 
    trans = trans.to(device)
    print('# parameters in Param2Image model:', get_n_params(trans))
    optim = torch.optim.Adam(trans.parameters(), lr=1e-3)#, weight_decay=1e-5)
    best_model = copy.deepcopy(trans)
    best_val_loss = 99999
    best_hasnt_changed_for = 0

    val_prop = 0.2

    train_strokes = strokes[int(val_prop*n):]
    train_parameters = parameters[int(val_prop*n):]

    val_strokes = strokes[:int(val_prop*n)]
    val_parameters = parameters[:int(val_prop*n)]
    print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))


    for it in tqdm(range(5000)):
        # with torch.autograd.set_detect_anomaly(True):
        if best_hasnt_changed_for >= 400 and it > 400:
            break # all done :)
        optim.zero_grad()

        pred_strokes = trans(train_parameters)

        loss = shift_invariant_loss(pred_strokes, train_strokes)
        # loss = nn.MSELoss()(pred_strokes, train_strokes) 

        ep_loss = loss.item()

        if not torch.isnan(loss):
            loss.backward()
            if it < 400:
                trans.renderer.weights.grad.data *= 0
            else:
                trans.renderer.weights.grad.data *= 0.1
            optim.step()
        else:
            print('Nan')
            break

        opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss, it)
        
        n_view = 10
        if it % 5 == 0:
            with torch.no_grad():
                trans.eval()
                pred_strokes_val = trans(val_parameters)

                # loss = shift_invariant_loss(pred_strokes_val, val_strokes)
                loss = nn.L1Loss()(pred_strokes_val, val_strokes)
                if it % 15 == 0: 
                    opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)

                    stroke_areas = val_strokes > 0.5
                    false_positive = (pred_strokes_val[~stroke_areas] > 0.5).sum() / stroke_areas.sum()
                    false_negative = (pred_strokes_val[stroke_areas] < 0.5).sum() / stroke_areas.sum()
                    opt.writer.add_scalar('loss/val_false_positive', false_positive.item(), it)
                    opt.writer.add_scalar('loss/val_false_negative', false_negative.item(), it)

                if loss.item() < best_val_loss and it > 50:
                    best_val_loss = loss.item()
                    best_hasnt_changed_for = 0
                    best_model = copy.deepcopy(trans)
                best_hasnt_changed_for += 5
                trans.train()

        if it % 20 == 0:
            # print(trans.y_m, trans.y_b)
            # print(trans.renderer.thicc_fact)
            # print(trans.renderer.weights)
            with torch.no_grad():
                def draw_grid(image, line_space_x=20, line_space_y=20):
                    H, W = image.shape
                    # image[0:H:line_space_x] = 0
                    # image[:, 0:W:line_space_y] = 0
                    return image
                best_model.eval()
                pred_strokes_val = best_model(val_parameters)
                real_imgs, pred_imgs = None, None
                for val_ind in range(min(n_view,len(val_strokes))):
                    pred_img = draw_grid(1-special_sigmoid(pred_strokes_val[val_ind]))
                    real_img = draw_grid(1-val_strokes[val_ind])
                    real_imgs = real_img if real_imgs is None else torch.cat([real_imgs, real_img], dim=0)
                    pred_imgs = pred_img if pred_imgs is None else torch.cat([pred_imgs, pred_img], dim=0)
                real_imgs[:,:5] = 0
                pred_imgs[:,:5] = 0
                whole_img = torch.cat([real_imgs, pred_imgs], dim=1)
                # whole_img = draw_grid(whole_img)
                opt.writer.add_image('images_stroke_modeling_stroke/val', process_img(whole_img), it)


                pred_strokes_train = best_model(train_parameters)
                real_imgs, pred_imgs = None, None
                for train_ind in range(min(n_view,len(train_strokes))):
                    pred_img = draw_grid(1-special_sigmoid(pred_strokes_train[train_ind]))
                    real_img = draw_grid(1-train_strokes[train_ind])
                    real_imgs = real_img if real_imgs is None else torch.cat([real_imgs, real_img], dim=0)
                    pred_imgs = pred_img if pred_imgs is None else torch.cat([pred_imgs, pred_img], dim=0)
                real_imgs[:,:5] = 0
                pred_imgs[:,:5] = 0
                whole_img = torch.cat([real_imgs, pred_imgs], dim=1)
                # whole_img = draw_grid(whole_img)
                opt.writer.add_image('images_stroke_modeling_stroke/train', process_img(whole_img), it)
                
    torch.save(best_model.cpu().state_dict(), os.path.join(opt.cache_dir, 'param2img.pt'))

    return h_og, w_og


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
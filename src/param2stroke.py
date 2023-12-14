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
        res_to_render = transforms.Resize((h_p2i_render_pix, w_p2i_render_pix), bicubic, antialias=True)

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
    return forward#param2img#param2imgs, resize



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


def to_full_param(length, bend, z, alpha=0.0, device='cuda'):
    full_param = torch.zeros((1,4)).to(device)
    
    full_param[0,0] = length 
    full_param[0,1] = bend 
    full_param[0,2] = z 
    full_param[0,3] = alpha

    return full_param

def process_img(img):
    return np.clip(img.detach().cpu().numpy(), a_min=0, a_max=1)*255

def log_all_permutations(model, writer, opt):
    # Length v Bend
    n_img = 5
    lengths = torch.linspace(opt.MIN_STROKE_LENGTH, opt.MAX_STROKE_LENGTH, steps=n_img)
    bends = torch.linspace(-1.0*opt.MAX_BEND, opt.MAX_BEND, steps=n_img)
    zs = torch.linspace(opt.MIN_STROKE_Z, 1.0, steps=n_img)
    alphas = torch.linspace(-1.*opt.MAX_ALPHA, opt.MAX_ALPHA, steps=n_img)


    whole_thing = []
    for i in range(n_img):
        row = []
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], bends[j], 0.5)
            s = 1-special_sigmoid(model(trajectory))
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
            row.append(s)
        whole_thing.append(np.concatenate(row, axis=1))
    whole_thing = np.concatenate(whole_thing, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(10,12))    
    ax.imshow(whole_thing, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    writer.add_figure('images_stroke_modeling/bend_vs_length', fig, 0)


    whole_thing = []
    for i in range(n_img):
        row = []
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], 0.0, zs[j])
            s = 1-special_sigmoid(model(trajectory))
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
            row.append(s)
        whole_thing.append(np.concatenate(row, axis=1))
    whole_thing = np.concatenate(whole_thing, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(10,12))    
    ax.imshow(whole_thing, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    writer.add_figure('images_stroke_modeling/thickness_vs_length', fig, 0)

    whole_thing = []
    for i in range(n_img):
        row = []
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], 0.0, 0.5, alphas[j])
            s = 1-model(trajectory)
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
            row.append(s)
        whole_thing.append(np.concatenate(row, axis=1))
    whole_thing = np.concatenate(whole_thing, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(10,12))    
    ax.imshow(whole_thing, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    writer.add_figure('images_stroke_modeling/length_vs_alpha', fig, 0)

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
    def __init__(self):
        super(StrokeParametersToImage, self).__init__()
        nh = 20
        self.nc = 20
        self.size_x = 128
        self.size_y = 64
        self.main = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, nh),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(nh),
            nn.Linear(nh, self.size_x*self.size_y),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, self.nc, kernel_size=5, padding='same', dilation=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.nc),
            nn.Conv2d(self.nc, 1, kernel_size=5, padding='same', dilation=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv((self.main(x).view(-1, 1, self.size_y, self.size_x)))[:,0]
        return x
    


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

    strokes = None
    for stroke_img_fn in stroke_img_fns:
        with gzip.GzipFile(stroke_img_fn,'r') as f:
            s = np.load(f, allow_pickle=True).astype(np.float32)/255.

            h_og, w_og = s[0].shape[0], s[0].shape[1]
            # Crop so that we only predict around the area around the stroke not all the background blank pix
            xtra_room_horz_m = 0.01 # Added padding to ensure we don't crop out actual paint
            xtra_room_vert_m = 0.001
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
    
    # with gzip.GzipFile(os.path.join(opt.cache_dir, 'stroke_intensities.npy'),'r') as f:
    #     strokes = np.load(f).astype(np.float32)/255.
    # parameters = np.load(os.path.join(opt.cache_dir, 'stroke_parameters.npy'), 
    #         allow_pickle=True, encoding='bytes') 

    strokes = torch.from_numpy(strokes).float().nan_to_num()
    parameters = torch.from_numpy(parameters.astype(np.float32)).float().nan_to_num()
    
    n = len(strokes)

    # Randomize
    rand_ind = torch.randperm(strokes.shape[0])
    strokes = strokes[rand_ind]
    parameters = parameters[rand_ind]

    # Discrete. Makes the model push towards making bolder strokes
    strokes[strokes >= 0.5] = 1.
    strokes[strokes < 0.5] = 0. # Make sure background is very much gone


    

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


    strokes = remove_background_noise(strokes)

    # Resize strokes the size they'll be predicted at
    t = StrokeParametersToImage()
    strokes = transforms.Resize((t.size_y,t.size_x), bicubic, antialias=True)(strokes)

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

    h, w = strokes[0].shape[0], strokes[0].shape[1]

    strokes = strokes.to(device)
    parameters = parameters.to(device)

    trans = StrokeParametersToImage() 
    trans = trans.to(device)
    print('# parameters in Param2Image model:', get_n_params(trans))
    optim = torch.optim.Adam(trans.parameters(), lr=1e-3)#, weight_decay=1e-5)
    best_model = copy.deepcopy(trans)
    best_val_loss = 99999
    best_hasnt_changed_for = 0

    val_prop = .2

    train_strokes = strokes[int(val_prop*n):]
    train_parameters = parameters[int(val_prop*n):]

    val_strokes = strokes[:int(val_prop*n)]
    val_parameters = parameters[:int(val_prop*n)]
    print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))

    param_stds = train_parameters.std(dim=0)

    for it in tqdm(range(5000)):
        if best_hasnt_changed_for >= 200 and it > 200:
            break # all done :)
        optim.zero_grad()

        noise = torch.randn(train_parameters.shape).to(device)*param_stds[None,:]*0.15 # For robustness
        pred_strokes = trans(train_parameters + noise)

        loss = shift_invariant_loss(pred_strokes, train_strokes)
        # loss = nn.MSELoss()(pred_strokes, train_strokes) # MSE loss produces crisper stroke images

        ep_loss = loss.item()
        loss.backward()
        optim.step()

        opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss, it)
        
        n_view = 10
        if it % 5 == 0:
            with torch.no_grad():
                trans.eval()
                pred_strokes_val = trans(val_parameters)

                loss = shift_invariant_loss(pred_strokes_val, val_strokes)
                if it % 15 == 0: 
                    opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)
                if loss.item() < best_val_loss and it > 50:
                    best_val_loss = loss.item()
                    best_hasnt_changed_for = 0
                    best_model = copy.deepcopy(trans)
                best_hasnt_changed_for += 5
                trans.train()


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
            l, b, z, alpha = val_parameters[val_ind][0], val_parameters[val_ind][1], val_parameters[val_ind][2], val_parameters[val_ind][3]
            # log_images([process_img(1-val_strokes[val_ind]),
            #     process_img(1-special_sigmoid(pred_strokes_val[val_ind]))], 
            #     ['real','sim'], 'images_stroke_modeling_stroke/val_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}_alph{:.2f}'.format(
            #                 val_ind, b, l, z, alpha), opt.writer)
            pred_img = draw_grid(1-special_sigmoid(pred_strokes_val[val_ind]))
            real_img = draw_grid(1-val_strokes[val_ind])
            real_imgs = real_img if real_imgs is None else torch.cat([real_imgs, real_img], dim=0)
            pred_imgs = pred_img if pred_imgs is None else torch.cat([pred_imgs, pred_img], dim=0)
        real_imgs[:,:5] = 0
        pred_imgs[:,:5] = 0
        whole_img = torch.cat([real_imgs, pred_imgs], dim=1)
        # whole_img = draw_grid(whole_img)
        opt.writer.add_image('images_stroke_modeling_stroke/val', process_img(whole_img), 0)


        pred_strokes_train = best_model(train_parameters)
        real_imgs, pred_imgs = None, None
        for train_ind in range(min(n_view,len(train_strokes))):
            l, b, z, alpha = train_parameters[train_ind][0], train_parameters[train_ind][1], train_parameters[train_ind][2], train_parameters[train_ind][3]
            # log_images([process_img(1-train_strokes[train_ind]),
            #     process_img(1-special_sigmoid(pred_strokes_train[train_ind]))], 
            #     ['real','sim'], 'images_stroke_modeling_stroke/train_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}_alph{:.2f}'.format(
            #                 train_ind, b, l, z, alpha), opt.writer)
            pred_img = draw_grid(1-special_sigmoid(pred_strokes_train[train_ind]))
            real_img = draw_grid(1-train_strokes[train_ind])
            real_imgs = real_img if real_imgs is None else torch.cat([real_imgs, real_img], dim=0)
            pred_imgs = pred_img if pred_imgs is None else torch.cat([pred_imgs, pred_img], dim=0)
        real_imgs[:,:5] = 0
        pred_imgs[:,:5] = 0
        whole_img = torch.cat([real_imgs, pred_imgs], dim=1)
        # whole_img = draw_grid(whole_img)
        opt.writer.add_image('images_stroke_modeling_stroke/train', process_img(whole_img), 0)
        
        log_all_permutations(best_model, opt.writer, opt)
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
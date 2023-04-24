# import pickle
import numpy as np
import torch
from torchvision import transforms
from torch import nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import requests
# from PIL import Image
import io
# import cv2
from tqdm import tqdm
import os
# import PIL.Image, PIL.ImageDraw
# from io import BytesIO
# import lpips
import copy
# import colour
# import random
import gzip
# import kornia as K
# import datetime
from torchvision.transforms.functional import affine

from options import Options
from my_tensorboard import TensorBoard


opt = Options()
opt.gather_options()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

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


def to_full_param(length, bend, z, alpha=0.0):
    full_param = torch.zeros((1,16)).to(device)
    
    # X
    full_param[0,0] = 0
    full_param[0,4] = length/3 
    full_param[0,8] = 2*length/3
    full_param[0,12] = length
    # Y
    full_param[0,1] = 0
    full_param[0,5] = bend
    full_param[0,9] = bend
    full_param[0,13] = 0
    # Z
    full_param[0,2] = 0.2
    full_param[0,6] = z
    full_param[0,10] = z
    full_param[0,14] = 0.2
    # alpha
    full_param[0,3] = alpha
    full_param[0,7] = alpha
    full_param[0,11] = alpha
    full_param[0,15] = alpha

    return full_param

def process_img(img):
    return np.clip(img.detach().cpu().numpy(), a_min=0, a_max=1)*255

def log_all_permutations(model, writer):
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
    writer.add_figure('images_stroke_modeling/bend_vs_length', fig, 0)


    whole_thing = []
    for i in range(n_img):
        row = []
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], 0.0, zs[j])
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
    def __init__(self, h, w):
        super(StrokeParametersToImage, self).__init__()
        nh = 20
        self.nc = 20
        self.size = 48#64
        self.size_x = 64#48
        self.size_y = 32#24
        self.main = nn.Sequential(
            nn.BatchNorm1d(16),
            #
            nn.Linear(16, nh),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(nh),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm1d(nh),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm1d(nh),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm1d(nh),
            # nn.Linear(nh, nh),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.Linear(nh, 64*64),
            nn.Linear(nh, self.size_x*self.size_y),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(1, self.nc, kernel_size=5, padding='same', dilation=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(self.nc),
            nn.Conv2d(self.nc, 1, kernel_size=5, padding='same', dilation=1),
            nn.Sigmoid()
        )
        self.w = w 
        self.h = h 
        self.res = transforms.Resize((self.h,self.w))
        self.res2 = transforms.Resize((self.h,self.w))

    def forward(self, x):
        # return self.fc1(x).view(-1, self.h, self.w)
        # return nn.Sigmoid()(self.fc4(nn.ReLU()(self.fc3(nn.ReLU()(self.fc2(nn.ReLU()(self.fc1(x))))))).view(-1, h, w))
        # return self.conv(self.main(x).view(-1, self.nc, self.h, self.w))[:,0]
        # x = self.res2(self.conv((self.main(x).view(-1, 1, 64, 64))))[:,0]
        x = self.res2(self.conv((self.main(x).view(-1, 1, self.size_y, self.size_x))))[:,0]
        # x = self.res2(self.conv((self.main(x).view(-1, 1, 32, 32))))[:,0]
        # x = 1/(1+torch.exp(-1.*(x*2-1) / 0.05))
        return x

    def forward_full(self, x):
        x = self.forward(x)
        return transforms.Pad((ws, hs, w_og-we, h_og-he))(x)
# class StrokeParametersToImage(nn.Module):
#     def __init__(self, h, w):
#         super(StrokeParametersToImage, self).__init__()
#         nh = 5
#         self.nc = 10

#         self.size_x = 48#64#128
#         self.size_y = 16#32#64
#         self.main = nn.Sequential(
#             nn.BatchNorm1d(16),
#             nn.Linear(16, nh),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.BatchNorm1d(nh),
#             nn.Linear(nh, self.size_x*self.size_y),
#             # nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv = nn.Sequential(
#             # nn.Conv2d(1, self.nc, kernel_size=5, padding='same', dilation=1),
#             # nn.LeakyReLU(0.2, inplace=True),
#             # nn.BatchNorm2d(self.nc),
#             # nn.Conv2d(self.nc, 1, kernel_size=5, padding='same', dilation=1),
#             nn.Sigmoid()
#         )
#         self.w = w 
#         self.h = h 
#         self.res = transforms.Resize((self.h,self.w))
#         self.res2 = transforms.Resize((self.h,self.w))

#     def forward(self, x):
#         x = self.res2(self.conv((self.main(x).view(-1, 1, self.size_y, self.size_x))))[:,0]
#         return x

class FillInParametersToImage(nn.Module):
    def __init__(self, h, w):
        super(FillInParametersToImage, self).__init__()
        nh = 3
        self.nc = 10

        self.size_x = 128#32#64#48
        self.size_y = 64#16#32#24
        self.main = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Linear(3, nh),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(nh),
            nn.Linear(nh, self.size_x*self.size_y),
            # nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = nn.Sequential(
            # nn.Conv2d(1, self.nc, kernel_size=5, padding='same', dilation=1),
            # nn.LeakyReLU(0.2, inplace=True),
            # nn.BatchNorm2d(self.nc),
            # nn.Conv2d(self.nc, 1, kernel_size=5, padding='same', dilation=1),
            nn.Sigmoid()
        )
        self.w = w 
        self.h = h 
        self.res = transforms.Resize((self.h,self.w))
        self.res2 = transforms.Resize((self.h,self.w))

    def forward(self, x):
        x = self.res2(self.conv((self.main(x).view(-1, 1, self.size_y, self.size_x))))[:,0]
        return x

l1_loss = nn.L1Loss()
def shift_invariant_loss(pred, real, n=8, delta=0.15):
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

def train_param2stroke(opt, is_brush_stroke=True):
    with gzip.GzipFile(os.path.join(opt.cache_dir, 
            'extended_stroke_library_intensities.npy' if is_brush_stroke else 'fill_in_intensities.npy'),'r') as f:
        strokes = np.load(f).astype(np.float32)/255.
    trajectories = np.load(os.path.join(opt.cache_dir, 
            'extended_stroke_library_trajectories.npy' if is_brush_stroke else 'fill_in_trajectories.npy'), 
            allow_pickle=True, encoding='bytes') 

    strokes = torch.from_numpy(strokes).float().nan_to_num()
    trajectories = torch.from_numpy(trajectories.astype(np.float32)).float().nan_to_num()
    
    n = len(strokes)
    
    # scale_factor = opt.max_height / strokes.shape[1]
    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    # strokes = transforms.Resize((int(strokes.shape[1]*scale_factor), int(strokes.shape[2]*scale_factor)))(strokes)
    strokes = transforms.Resize((h,w))(strokes)

    # Randomize
    rand_ind = torch.randperm(strokes.shape[0])
    strokes = strokes[rand_ind]
    trajectories = trajectories[rand_ind]

    # Discrete. Makes the model push towards making bolder strokes
    # strokes[strokes >= 0.5] = 1.
    # strokes[strokes < 0.5] = 0.

    # h, w = strokes[0].shape[0], strokes[0].shape[1]
    h_og, w_og = h, w

    # Crop
    if is_brush_stroke:
        hs, he = int(.4*h), int(0.6*h)
        ws, we = int(0.45*w), int(0.75*w)
    else:
        # Fill in needs to leave more
        hs, he = int(.4*h), int(0.6*h)
        ws, we = int(0.45*w), int(0.75*w)

    strokes = strokes[:, hs:he, ws:we]

    for i in range(len(strokes)):
        strokes[i] -= strokes[i].min()
        if strokes[i].max() > 0.01:
            strokes[i] /= strokes[i].max()
        # strokes[i] *= 0.95
    
    # Filter out strokes that are bad perception. Avg is too high.
    # One bad apple can really spoil the bunch
    good_strokes = []
    good_traj = []
    for i in range(len(strokes)):
        if strokes[i].mean() < 0.4: 
            good_strokes.append(strokes[i])
            good_traj.append(trajectories[i])
    print(len(strokes)- len(good_strokes), 'strokes removed because average value too high')
    strokes = torch.stack(good_strokes, dim=0)
    trajectories = torch.stack(good_traj, dim=0)


    h, w = strokes[0].shape[0], strokes[0].shape[1]
    ##########FOR SPEED#########################################
    scale_factor = 4
    strokes = transforms.Resize((int(h/scale_factor), int(w/scale_factor)))(strokes) 
    h, w = strokes[0].shape[0], strokes[0].shape[1]

    strokes = strokes.to(device)
    trajectories = trajectories.to(device)

    for model_ind in range(opt.n_stroke_models):
        trans = StrokeParametersToImage(h,w) if is_brush_stroke else FillInParametersToImage(h,w)
        trans = trans.to(device)
        print('# parameters in Param2Image model:', get_n_params(trans))
        optim = torch.optim.Adam(trans.parameters(), lr=1e-3)#, weight_decay=1e-5)
        best_model = copy.deepcopy(trans)
        best_val_loss = 99999
        best_hasnt_changed_for = 0

        val_prop = .2

        train_strokes = strokes[int(val_prop*n):]
        train_trajectories = trajectories[int(val_prop*n):]

        val_strokes = strokes[:int(val_prop*n)]
        val_trajectories = trajectories[:int(val_prop*n)]
        print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))

        for it in tqdm(range(20000)):
            # if it == 20:
            #     best_model=trans
            #     break
            if best_hasnt_changed_for >= 200 and it > 200:
                break # all done :)
            optim.zero_grad()

            noise = torch.randn(train_trajectories.shape).to(device)*0#*0.001 # For robustness
            pred_strokes = trans(train_trajectories + noise)

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
                    pred_strokes_val = trans(val_trajectories)

                    loss = shift_invariant_loss(pred_strokes_val, val_strokes)
                    if it % 15 == 0: 
                        opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)
                    if loss.item() < best_val_loss and it > 50:
                        best_val_loss = loss.item()
                        best_hasnt_changed_for = 0
                        best_model = copy.deepcopy(trans)
                    best_hasnt_changed_for += 5
                    trans.train()

        if model_ind == 0:
            if is_brush_stroke:
                with torch.no_grad():
                    def draw_grid(image, line_space_x=20, line_space_y=20):
                        H, W = image.shape
                        image[0:H:line_space_x] = 0
                        image[:, 0:W:line_space_y] = 0
                        return image
                    best_model.eval()
                    pred_strokes_val = best_model(val_trajectories)
                    real_imgs, pred_imgs = None, None
                    for val_ind in range(min(n_view,len(val_strokes))):
                        b, l, z, alpha = val_trajectories[val_ind][5], val_trajectories[val_ind][12], val_trajectories[val_ind][6], val_trajectories[val_ind][-1]
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


                    pred_strokes_train = best_model(train_trajectories)
                    real_imgs, pred_imgs = None, None
                    for train_ind in range(min(n_view,len(train_strokes))):
                        b, l, z, alpha = train_trajectories[train_ind][5], train_trajectories[train_ind][12], train_trajectories[train_ind][6], train_trajectories[train_ind][-1]
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
            else:
                with torch.no_grad():
                    best_model.eval()
                    pred_strokes_val = best_model(val_trajectories)
                    for val_ind in range(min(n_view,len(val_strokes))):
                        # print(val_trajectories[val_ind])
                        h0, h1, length = val_trajectories[val_ind][0], val_trajectories[val_ind][1], val_trajectories[val_ind][2]
                        log_images([process_img(1-val_strokes[val_ind]),
                            process_img(1-special_sigmoid(pred_strokes_val[val_ind]))], 
                            ['real','sim'], 'images_stroke_modeling_fill_in/val_{}_sim_stroke_best_h0{:.2f}_h1{:.2f}_l{:.2f}'.format(
                                        val_ind, h0, h1, length), opt.writer)

                    pred_strokes_train = best_model(train_trajectories)
                    for train_ind in range(min(n_view,len(train_strokes))):
                        h0, h1, length = train_trajectories[train_ind][0], train_trajectories[train_ind][1], train_trajectories[train_ind][2]
                        log_images([process_img(1-train_strokes[train_ind]),
                            process_img(1-special_sigmoid(pred_strokes_train[train_ind]))], 
                            ['real','sim'], 'images_stroke_modeling_fill_in/train_{}_sim_stroke_best_h0{:.2f}_h1{:.2f}_l{:.2f}'.format(
                                        train_ind, h0, h1, length), opt.writer)
            if is_brush_stroke:
                log_all_permutations(best_model, opt.writer)
        torch.save(best_model.cpu().state_dict(), os.path.join(opt.cache_dir, 'param2img{}{}.pt'.format('' if is_brush_stroke else '_fill_in', model_ind)))

    return h_og, w_og

def n_stroke_test(opt):
    #strokes = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy')).astype(np.float32)/255.
    with gzip.GzipFile(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy'),'r') as f:
        strokes = np.load(f).astype(np.float32)/255.
    trajectories = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_trajectories.npy'), 
            allow_pickle=True, encoding='bytes') 

    strokes = torch.from_numpy(strokes).to(device).float().nan_to_num()
    trajectories = torch.from_numpy(trajectories.astype(np.float32)).to(device).float().nan_to_num()

    # Randomize
    rand_ind = torch.randperm(strokes.shape[0])
    strokes = strokes[rand_ind]
    trajectories = trajectories[rand_ind]

    stroke_shape = np.load(os.path.join(opt.cache_dir, 'stroke_size.npy'))
    h, w = stroke_shape[0], stroke_shape[1]
    strokes = transforms.Resize((h,w))(strokes)

    h_og, w_og = h, w

    # Crop
    hs, he = int(.4*h), int(0.6*h)
    ws, we = int(0.45*w), int(0.75*w)
    strokes = strokes[:, hs:he, ws:we]




    n = len(strokes)
    print(n)
    test_prop = 0.2 
    test_strokes = strokes[:int(test_prop*n)]
    test_trajectories = trajectories[:int(test_prop*n)]
    strokes = strokes[int(test_prop*n):]
    trajectories = trajectories[int(test_prop*n):]
    n = len(strokes)
    print(n, 'test strokes', len(test_strokes), len(strokes))

    disp_traj = to_full_param(0.04, 0.02, .5)



    h, w = strokes[0].shape[0], strokes[0].shape[1]

    n = len(strokes)

    plot_x = []
    plot_y = []

    plot_x_std = []
    plot_y_std = []
    
    for n_strokes in range(15, n, 5):
        s = strokes[:n_strokes]
        t = trajectories[:n_strokes]



        avg_test_err = 0
        test_err = []
        n_folds = 5
        for fold in range(n_folds):
            trans = StrokeParametersToImage(h,w).to(device)
            #print('# parameters in StrokeParam2Image model:', get_n_params(trans))
            optim = torch.optim.Adam(trans.parameters(), lr=1e-3)
            best_model = None
            best_val_loss = 999
            best_hasnt_changed_for = 0

            #val_prop = .3

            train_strokes = torch.cat([s[:int((fold/n_folds)*n_strokes)], s[int(((fold+1)/n_folds)*n_strokes):]], dim=0)
            train_trajectories = torch.cat([t[:int((fold/n_folds)*n_strokes)], t[int(((fold+1)/n_folds)*n_strokes):]], dim=0)
            val_strokes = s[int((fold/n_folds)*n_strokes):int(((fold+1)/n_folds)*n_strokes)]
            val_trajectories = t[int((fold/n_folds)*n_strokes):int(((fold+1)/n_folds)*n_strokes)]
            if fold == 0:
                print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))

            for it in tqdm(range(2000)):
                if best_hasnt_changed_for >= 200:
                    break # all done :)
                optim.zero_grad()

                noise = torch.randn(train_trajectories.shape).to(device)*0.005 # For robustness
                pred_strokes = trans(train_trajectories + noise)

                loss = loss_fcn(pred_strokes, train_strokes)
                # loss = nn.MSELoss()(pred_strokes, train_strokes) # MSE loss produces crisper stroke images

                ep_loss = loss.item()
                loss.backward()
                optim.step()

                opt.writer.add_scalar('loss/train_loss_stroke_model', ep_loss, it)
                
                n_view = 10
                with torch.no_grad():
                    trans.eval()
                    pred_strokes_val = trans(val_trajectories)
                    # loss = nn.MSELoss()(pred_strokes_val, val_strokes)
                    loss = loss_fcn(pred_strokes_val, val_strokes)
                    # print(pred_strokes_val.shape, val_strokes.shape)
                    opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)

                    if loss.item() < best_val_loss and it > 50:
                        best_val_loss = loss.item()
                        best_hasnt_changed_for = 0
                        best_model = copy.deepcopy(trans)
                    best_hasnt_changed_for += 1
                    trans.train()
            with torch.no_grad():
                trans.eval()
                pred_strokes_test = best_model(test_trajectories)
                # loss = nn.MSELoss()(pred_strokes_test, test_strokes)
                test_loss = loss_fcn(pred_strokes_test, test_strokes)
                
                trans.train()

            with torch.no_grad():
                trans.eval()# The display model
                import matplotlib.pyplot as plt 
                fig, ax = plt.subplots(1)
                disp_stroke = 1 - best_model(disp_traj)
                disp_stroke = np.clip(disp_stroke.detach().cpu().numpy()[0], a_min=0, a_max=1)
                ax.imshow(disp_stroke, cmap='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                fig.tight_layout()
                writer.add_figure('limit_data_stroke_model/n_strokes{}'.format(n_strokes), fig, fold)
                trans.train()
            avg_test_err += test_loss.item()/n_folds
            test_err.append(test_loss.item())
        print('test error', avg_test_err)
        opt.writer.add_scalar('loss/test_loss_stroke_model', avg_test_err, n_strokes)
        test_err_std = np.array(test_err).std()
        opt.writer.add_scalar('loss/test_loss_std_stroke_model', test_err_std, n_strokes)
        
        plot_x.append(n_strokes)
        plot_y.append(avg_test_err)
        plot_x_std.append(n_strokes)
        plot_y_std.append(test_err_std)

        import matplotlib.pyplot as plt 
        plt.rcParams["font.family"] = "Times New Roman"
        # plot_x = [1,2,3]
        # plot_y = [.1,.05, .01]
        plt.plot(plot_x, plot_y)
        csfont = {'fontname':'Times New Roman'}
        plt.title('Stroke Shape Model Performance Versus Number of Training Examples')
        plt.ylabel('Average Absolute Test Error')
        plt.xlabel('Number of Training and Validation Brush Strokes')
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.savefig('err_v_strokes.svg', format='svg')
        plt.show()

        fig, ax = plt.subplots()

        ax.plot(plot_x, plot_y, color='red')
        ax.set_yscale('log')
        ax.tick_params(axis='y', labelcolor='red')

        ax2 = ax.twinx()
        ax2.plot(plot_x_std, plot_y_std, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.tick_params(axis='y', labelcolor='red')
        plt.savefig('err_v_strokes_with_std.svg', format='svg')
        plt.show()


        fig, ax = plt.subplots()

        ax.plot(plot_x, plot_y, 'brown')
        ax.fill_between(plot_x, np.array(plot_y) - np.array(plot_y_std), np.array(plot_y) + np.array(plot_y_std),
            color='lightsalmon')
        ax.set_yscale('log')
        ax.set_title('Stroke Shape Model Performance Versus Number of Training Examples')
        ax.set_ylabel('Log Average Absolute Test Error')
        ax.set_xlabel('Number of Training and Validation Brush Strokes')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax2.tick_params(axis='y', labelcolor='green')
        # ax.tick_params(axis='y', labelcolor='red')
        plt.savefig('err_v_strokes_with_std2.svg', format='svg')
        plt.show()


    return h_og, w_og

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    torch.manual_seed(0)
    np.random.seed(0)

    # b = './painting'
    # all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
    # tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
    # if '_planner' not in tensorboard_dir:
    #     tensorboard_dir += '_planner'
    # writer = TensorBoard(tensorboard_dir)
    # opt.writer = writer

    from paint_utils3 import create_tensorboard
    opt.writer = create_tensorboard()

    # n_stroke_test(opt)

    # Train fill-in
    # train_param2stroke(opt, is_brush_stroke=False)

    # Train brush strokes
    train_param2stroke(opt)
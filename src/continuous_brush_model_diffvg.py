# import pickle
import numpy as np
import torch
from torchvision import transforms
from torch import nn

import matplotlib
# matplotlib.use('Agg')
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
import datetime
from tqdm import tqdm

from options import Options
from tensorboard import TensorBoard

# import sys
# sys.path.append('C:/Users/Peter/paint/sawyerpainter2/diffvg')
# sys.path.append('C:/Users/Peter/paint/sawyerpainter2/diffvg/pydiffvg')
# sys.path.append('/mnt/c/Users/Peter/paint/sawyerpainter2/diffvg')
# sys.path.append('/mnt/c/Users/Peter/paint/sawyerpainter2/diffvg/pydiffvg')
import pydiffvg

def render_drawing(shapes, shape_groups,\
                   canvas_width, canvas_height, n_iter, save=False):
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, n_iter, None, *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])


    img = img[:, :, :3]
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
    return img

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

def init_diffvg_brush_stroke(h, w):

    cp = torch.tensor([
        [.5*w, .5*h],
        [.5*w+.01*w, .5*h],
        [.5*w+.02*w, .5*h],
        [.5*w+.03*w, .5*h],
    ])
    shape = pydiffvg.Path(
            num_control_points=torch.tensor([2], dtype = torch.int32),
            points=cp,
            stroke_width=torch.tensor(2.),
            is_closed=False
    )
    shape_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=None,
            stroke_color=torch.tensor([.05,.05,.05, .95]))
    return shape, shape_group


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if not torch.cuda.is_available():
    print('Using CPU..... good luck')

def special_sigmoid(x):
    # return 1/(1+torch.exp(-1.*((x*2-1)+0.2) / 0.05))
    # return x

    # x[x < 0.1] = 1/(1+torch.exp(-1.*((x[x < 0.1]*2-1)+0.2) / 0.05))
    return x

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn 
    return pp

def to_full_param(length, bend, z):
    full_param = torch.zeros((1,12)).to(device)
    
    # X
    full_param[0,0] = 0
    full_param[0,3] = length/3 
    full_param[0,6] = 2*length/3
    full_param[0,9] = length
    # Y
    full_param[0,1] = 0
    full_param[0,4] = bend
    full_param[0,7] = bend
    full_param[0,10] = 0
    # Z
    full_param[0,2] = 0.2
    full_param[0,5] = z
    full_param[0,8] = z
    full_param[0,11] = 0.2

    return full_param

def process_img(img):
    return np.clip(img.detach().cpu().numpy(), a_min=0, a_max=1)*255

def log_all_permutations(model, writer):
    # Length v Bend
    n_img = 7
    lengths = torch.arange(n_img)/(n_img-1)*(0.05-0.01) + 0.01
    bends = torch.arange(n_img)/(n_img-1)*0.04 - 0.02
    zs = torch.arange(n_img)/(n_img-1)

    fig, ax = plt.subplots(n_img, n_img, figsize=(10,12))

    for i in range(n_img):
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], bends[j], 0.5)
            s = 1-model(trajectory)
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

            ax[i,j].imshow(s, cmap='gray')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    fig.tight_layout()
    # plt.show()
    writer.add_figure('images_stroke_modeling/stroke_modelling_bend_vs_length', fig, 0)

    fig, ax = plt.subplots(n_img, n_img, figsize=(10,12))

    for i in range(n_img):
        for j in range(n_img):
            trajectory = to_full_param(lengths[i], 0.0, zs[j])
            s = 1-model(trajectory)
            # print(s.shape)
            s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

            ax[i,j].imshow(s, cmap='gray')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
    fig.tight_layout()
    writer.add_figure('images_stroke_modeling/stroke_modelling_thickness_vs_length', fig, 0)
    # plt.show()

    img_fig = None
    for i in range(n_img):
        trajectory = to_full_param(.04, 0.0, zs[i])
        s = 1-model(trajectory)
        # print(s.shape)
        s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
        s = np.pad(s, (3), 'constant', constant_values=(1.))
        if img_fig is None:
            img_fig = s 
        else:
            img_fig = np.concatenate([img_fig, s], axis=1)
    plt.figure(figsize=(19,3))
    plt.imshow(img_fig, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.savefig('thickness.png')
    # plt.show()

    img_fig = None
    for i in range(n_img):
        trajectory = to_full_param(.04, bends[i], .5)
        s = 1-model(trajectory)
        # print(s.shape)
        s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
        s = np.pad(s, (3), 'constant', constant_values=(1.))
        if img_fig is None:
            img_fig = s 
        else:
            img_fig = np.concatenate([img_fig, s], axis=1)
    plt.figure(figsize=(19,3))
    plt.imshow(img_fig, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.savefig('bends.png')
    # plt.show()

    img_fig = None
    for i in range(n_img):
        trajectory = to_full_param(lengths[i], 0, .5)
        s = 1-model(trajectory)
        # print(s.shape)
        s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)
        s = np.pad(s, (3), 'constant', constant_values=(1.))
        if img_fig is None:
            img_fig = s 
        else:
            img_fig = np.concatenate([img_fig, s], axis=1)
    plt.figure(figsize=(19,3))
    plt.imshow(img_fig, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.savefig('lengths.png')
    # plt.show()

    # fig, ax = plt.subplots(1, n_img, figsize=(12,1))
    # for i in range(n_img):
    #     trajectory = to_full_param(.04, 0.0, zs[i])
    #     s = 1-model(trajectory)
    #     # print(s.shape)
    #     s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

        

    #     ax[i].imshow(s, cmap='gray')
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])
    # fig.tight_layout()
    # plt.show()
    # writer.add_figure('images_stroke_modeling/stroke_modelling_thickness', fig, 0)

    # fig, ax = plt.subplots(1, n_img, figsize=(12,1))
    # for i in range(n_img):
    #     trajectory = to_full_param(.04, bends[i], .5)
    #     s = 1-model(trajectory)
    #     # print(s.shape)
    #     s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

    #     ax[i].imshow(s, cmap='gray')
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])
    # fig.tight_layout()
    # plt.show()
    # writer.add_figure('images_stroke_modeling/stroke_modelling_bend', fig, 0)

    # fig, ax = plt.subplots(1, n_img, figsize=(12,1))
    # for i in range(n_img):
    #     trajectory = to_full_param(lengths[i], 0, .5)
    #     s = 1-model(trajectory)
    #     # print(s.shape)
    #     s = np.clip(s.detach().cpu().numpy()[0], a_min=0, a_max=1)

    #     ax[i].imshow(s, cmap='gray')
    #     ax[i].set_xticks([])
    #     ax[i].set_yticks([])
    # fig.tight_layout()
    # plt.show()
    # writer.add_figure('images_stroke_modeling/stroke_modelling_length', fig, 0)


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

def log_image(img, title, label, writer, step=0):
    fig, ax = plt.subplots(1)

    # for i in range(len(imgs)):
    # print(imgs[i].min(), imgs[i].max())
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(os.path.join('stroke_comparison', '{}.png'.format(label.split('/')[-1])))
    writer.add_figure(label, fig, step)

class StrokeParametersToImage(nn.Module):
    def __init__(self, h, w):
        super(StrokeParametersToImage, self).__init__()
        nh = 10
        self.nc = 10
        self.main = nn.Sequential(
            nn.BatchNorm1d(12),
            #
            nn.Linear(12, nh),
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
            nn.Linear(nh, 48*48),
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
        x = self.res2(self.conv((self.main(x).view(-1, 1, 48, 48))))[:,0]
        # x = self.res2(self.conv((self.main(x).view(-1, 1, 32, 32))))[:,0]
        # x = 1/(1+torch.exp(-1.*(x*2-1) / 0.05))
        return x

    def forward_full(self, x):
        x = self.forward(x)
        return transforms.Pad((ws, hs, w_og-we, h_og-he))(x)

loss_fcn = nn.L1Loss()#nn.MSELoss()

def train_param2stroke(opt):
    #strokes = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy')).astype(np.float32)/255.
    with gzip.GzipFile(os.path.join(opt.cache_dir, 'extended_stroke_library_intensities.npy'),'r') as f:
        strokes = np.load(f).astype(np.float32)/255.
    trajectories = np.load(os.path.join(opt.cache_dir, 'extended_stroke_library_trajectories.npy'), 
            allow_pickle=True, encoding='bytes') 

    strokes = torch.from_numpy(strokes).to(device).float().nan_to_num()
    trajectories = torch.from_numpy(trajectories.astype(np.float32)).to(device).float().nan_to_num()
    
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
    hs, he = int(.4*h), int(0.6*h)
    ws, we = int(0.3*w), int(0.7*w)
    strokes = strokes[:, hs:he, ws:we]

    test_strokes = strokes[:90]
    test_trajectories = trajectories[:90]
    strokes = strokes[90:]
    trajectories = trajectories[90:]

    n = len(strokes)

    for i in range(len(strokes)):
        strokes[i] -= strokes[i].min()
        strokes[i] /= strokes[i].max()

    h, w = strokes[0].shape[0], strokes[0].shape[1]

    for model_ind in range(1):#range(opt.n_stroke_models):
        trans = StrokeParametersToImage(h,w).to(device)
        print('# parameters in StrokeParam2Image model:', get_n_params(trans))
        optim = torch.optim.Adam(trans.parameters(), lr=1e-3)
        best_model = copy.deepcopy(trans)
        best_val_loss = 999
        best_hasnt_changed_for = 0

        val_prop = .3

        train_strokes = strokes[int(val_prop*n):]
        train_trajectories = trajectories[int(val_prop*n):]
        val_strokes = strokes[:int(val_prop*n)]
        val_trajectories = trajectories[:int(val_prop*n)]
        print('{} training strokes. {} validation strokes'.format(len(train_strokes), len(val_strokes)))

        for it in tqdm(range(4000)):
            if best_hasnt_changed_for >= 400:
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
            
            n_view = 50
            with torch.no_grad():
                trans.eval()
                pred_strokes_val = trans(val_trajectories)
                # if it % 50 == 0:
                #     for val_ind in range(min(n_view,len(val_strokes))):
                #         log_images([process_img(val_strokes[val_ind]),
                #             process_img(pred_strokes_val[val_ind])], 
                #             ['real','sim'], 'images_stroke_modeling/val_{}_stroke'.format(val_ind), opt.writer, step=it)
                #     pred_strokes_train = trans(train_trajectories)
                #     for train_ind in range(min(n_view,len(train_strokes))):
                #         log_images([process_img(train_strokes[train_ind]),
                #             process_img(pred_strokes_train[train_ind])], 
                #             ['real','sim'], 'images_stroke_modeling/train_{}_stroke'.format(train_ind), opt.writer, step=it)

                # loss = nn.MSELoss()(pred_strokes_val, val_strokes)
                loss = loss_fcn(pred_strokes_val, val_strokes)
                opt.writer.add_scalar('loss/val_loss_stroke_model', loss.item(), it)
                if loss.item() < best_val_loss and it > 50:
                    best_val_loss = loss.item()
                    best_hasnt_changed_for = 0
                    best_model = copy.deepcopy(trans)
                best_hasnt_changed_for += 1
                trans.train()

        import glob
        files = glob.glob('./stroke_comparison/**')
        for f in files:
            os.remove(f)

        if model_ind == 0:
            # with torch.no_grad():
            best_model.eval()
            pred_strokes_test = best_model(test_trajectories)
            diff_differences = []
            frida_differences = []
            for test_ind in range(min(n_view,len(test_strokes))):
                # print(test_trajectories[test_ind])
                b, l, z = test_trajectories[test_ind][4], test_trajectories[test_ind][9], test_trajectories[test_ind][5]

                with torch.no_grad():
                    shape, shape_group = init_diffvg_brush_stroke(h, w)
                    shape.points[:,0] = (torch.linspace(0,1,4).to(device)*(l*100/(35.5*.4))+.5)*w#(real_stroke_params[s_ind].xs/opt.CANVAS_WIDTH+.5) * w
                    shape.points[:,1] = .5*h#(1-(real_stroke_params[s_ind].ys/opt.CANVAS_HEIGHT+.5)) * h
                    shape.points[1:3,1] = .5*h + -1*b*100/(28*.2)*h

                    shape.stroke_width = (z/1.3+0.2)*.05*w
                    diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)


                ####################### Optimize DiffVG stroke ##################
                points_vars = []
                color_vars = []
                width_vars = []

                shape.points.requires_grad = True
                points_vars.append(shape.points)
                shape.stroke_width.requires_grad = True
                width_vars.append(shape.stroke_width)
                shape_group.stroke_color.requires_grad = True
                color_vars.append(shape_group.stroke_color)

                points_optim = torch.optim.RMSprop(points_vars, lr=0.1*.05)
                width_optim = torch.optim.RMSprop(width_vars, lr=0.1*.05)
                color_optim = torch.optim.RMSprop(color_vars, lr=0.01*.05)

                for it in range(10):
                    points_optim.zero_grad(), 
                    color_optim.zero_grad()
                    width_optim.zero_grad()
                    diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)
                    loss = torch.nn.L1Loss()(diffvg_stroke.mean(dim=1)[0], 1-test_strokes[test_ind].detach())
                    # print(loss.item())
                    loss.backward()
                    # points_optim.step(), 
                    # color_optim.step()
                    width_optim.step()


                ##################################################




                # print(diffvg_stroke)
                # print(l)
                # print(shape.points)
                # show_img(diffvg_stroke)
                with torch.no_grad():
                    diff = diffvg_stroke.to(device).mean(dim=1)[0]
                    real = 1-test_strokes[test_ind]
                    ours = 1-special_sigmoid(pred_strokes_test[test_ind])

                with torch.no_grad():
                    diff_differences.append(torch.nn.L1Loss()(diff, real).item())
                    frida_differences.append(torch.nn.L1Loss()(ours, real).item())

                # Extra crop
                cl, cr, ct, cb = 0.45, 0.9, .9, 0.1
                ch, cw = diff.shape
                diff = diff[int(cb*ch):int(ct*ch), int(cl*cw):int(cr*cw)]
                real = real[int(cb*ch):int(ct*ch), int(cl*cw):int(cr*cw)]
                ours = ours[int(cb*ch):int(ct*ch), int(cl*cw):int(cr*cw)]

                img = torch.cat([real,diff,  
                    ours], dim=1)
                # show_img(img)
                # log_image(process_img(img), 'b={:.3f}  l={:.3f}  h={:.3f}'.format(b, l, z), 
                #     'images_stroke_modeling/test_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}'.format(test_ind, b, l, z), opt.writer)

            diff_differences = np.array(diff_differences)
            frida_differences = np.array(frida_differences)

            print('diff diffvg', diff_differences.mean(), len(diff_differences))
            print('diff frida ', frida_differences.mean(), len(frida_differences))

            from scipy.stats import ttest_ind
            print(ttest_ind(frida_differences, diff_differences,  
                equal_var=False, alternative='less'))


            diff_differences = []
            frida_differences = []
            best_model.eval()
            pred_strokes_train = best_model(train_trajectories)
            for train_ind in range(min(n_view,len(train_strokes))):
                # print(train_trajectories[train_ind])
                b, l, z = train_trajectories[train_ind][4], train_trajectories[train_ind][9], train_trajectories[train_ind][5]

                with torch.no_grad():
                    shape, shape_group = init_diffvg_brush_stroke(h, w)
                    shape.points[:,0] = (torch.linspace(0,1,4).to(device)*(l*100/(35.5*.4))+.5)*w#(real_stroke_params[s_ind].xs/opt.CANVAS_WIDTH+.5) * w
                    shape.points[:,1] = .5*h#(1-(real_stroke_params[s_ind].ys/opt.CANVAS_HEIGHT+.5)) * h
                    shape.points[1:3,1] = .5*h + -1*b*100/(28*.2)*h

                    shape.stroke_width = (z/1.3+0.2)*.05*w
                    diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)


                ####################### Optimize DiffVG stroke ##################
                points_vars = []
                color_vars = []
                width_vars = []

                shape.points.requires_grad = True
                points_vars.append(shape.points)
                shape.stroke_width.requires_grad = True
                width_vars.append(shape.stroke_width)
                shape_group.stroke_color.requires_grad = True
                color_vars.append(shape_group.stroke_color)

                points_optim = torch.optim.RMSprop(points_vars, lr=0.1*.05)
                width_optim = torch.optim.RMSprop(width_vars, lr=0.1*.05)
                color_optim = torch.optim.RMSprop(color_vars, lr=0.01*.05)

                for it in range(10):
                    points_optim.zero_grad(), 
                    color_optim.zero_grad()
                    width_optim.zero_grad()
                    diffvg_stroke = render_drawing([shape], [shape_group], w, h, 0)
                    loss = torch.nn.L1Loss()(diffvg_stroke.mean(dim=1)[0], 1-train_strokes[train_ind].detach())
                    # print(loss.item())
                    loss.backward()
                    # points_optim.step(), 
                    # color_optim.step()
                    width_optim.step()


                ##################################################

                # print(diffvg_stroke)
                # print(l)
                # print(shape.points)
                # show_img(diffvg_stroke)
                # diff = diffvg_stroke.to(device).mean(dim=1)[0]
                # real = 1-train_strokes[train_ind]
                # ours = 1-special_sigmoid(pred_strokes_train[train_ind])

                with torch.no_grad():
                    diff = diffvg_stroke.to(device).mean(dim=1)[0]
                    real = 1-train_strokes[train_ind]
                    ours = 1-special_sigmoid(pred_strokes_train[train_ind])

                with torch.no_grad():
                    diff_differences.append(torch.nn.L1Loss()(diff, real).item())
                    frida_differences.append(torch.nn.L1Loss()(ours, real).item())

                # Extra crop
                cl, cr, ct, cb = 0.45, 0.9, .9, 0.1
                ch, cw = diff.shape
                diff = diff[int(cb*ch):int(ct*ch), int(cl*cw):int(cr*cw)]
                real = real[int(cb*ch):int(ct*ch), int(cl*cw):int(cr*cw)]
                ours = ours[int(cb*ch):int(ct*ch), int(cl*cw):int(cr*cw)]

                img = torch.cat([real, diff, ours], dim=1)
                # show_img(img)
                log_image(process_img(img), 'b={:.3f}  l={:.3f}  h={:.3f}'.format(b, l, z), 
                    'images_stroke_modeling/train_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}'.format(train_ind, b, l, z), opt.writer)









                    # log_images([process_img(1-test_strokes[test_ind]),
                    #     process_img(1-special_sigmoid(pred_strokes_test[test_ind]))],
                    #     ['real','sim'], 'images_stroke_modeling/test_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}'.format(test_ind, b, l, z), opt.writer)



                # pred_strokes_train = best_model(train_trajectories)
                # for train_ind in range(min(n_view,len(train_strokes))):
                #     b, l, z = train_trajectories[train_ind][4], train_trajectories[train_ind][9], train_trajectories[train_ind][5]
                #     log_images([process_img(1-train_strokes[train_ind]),
                #         process_img(1-special_sigmoid(pred_strokes_train[train_ind]))], 
                #         ['real','sim'], 'images_stroke_modeling/train_{}_sim_stroke_best_b{:.2f}_l{:.2f}_z{:.2f}'.format(train_ind, b, l, z), opt.writer)

                # # Make them into full sized strokes
                # pred_strokes_test = best_model(test_trajectories)
                # pred_strokes_test = special_sigmoid(pred_strokes_test)
                # pred_strokes_test = transforms.Pad((ws, hs, w_og-we, h_og-he))(pred_strokes_test)
                # test_strokes_full = transforms.Pad((ws, hs, w_og-we, h_og-he))(test_strokes)
                # for test_ind in range(min(n_view,len(test_strokes))):
                #     log_images([process_img(1-test_strokes_full[test_ind]),
                #         process_img(1-pred_strokes_test[test_ind])],
                #         ['real','sim'], 'images_stroke_modeling/test_{}_stroke_full'.format(test_ind), opt.writer)
            
            diff_differences = np.array(diff_differences)
            frida_differences = np.array(frida_differences)
            print('training')
            print('diff diffvg', diff_differences.mean(), len(diff_differences))
            print('diff frida ', frida_differences.mean(), len(frida_differences))

            from scipy.stats import ttest_ind
            print(ttest_ind(frida_differences, diff_differences,  
                equal_var=False, alternative='less'))
            log_all_permutations(best_model, opt.writer)
        # torch.save(best_model.cpu().state_dict(), os.path.join(opt.cache_dir, 'param2img{}.pt'.format(model_ind)))

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


    h, w = strokes[0].shape[0], strokes[0].shape[1]

    n = len(strokes)

    plot_x = []
    plot_y = []
    
    for n_strokes in range(10, n, 5):
        s = strokes[:n_strokes]
        t = trajectories[:n_strokes]


        avg_test_err = 0
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
            avg_test_err += test_loss.item()/n_folds
        print('test error', avg_test_err)
        opt.writer.add_scalar('loss/test_loss_stroke_model', avg_test_err, n_strokes)

        plot_x.append(n_strokes)
        plot_y.append(avg_test_err)

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
    # plt.savefig('err_v_strokes.svg', format='svg')
    plt.show()


    return h_og, w_og

if __name__ == '__main__':
    global opt, strokes
    opt = Options()
    opt.gather_options()

    # b = './painting'
    # all_subdirs = [os.path.join(b, d) for d in os.listdir(b) if os.path.isdir(os.path.join(b, d))]
    # tensorboard_dir = max(all_subdirs, key=os.path.getmtime) # most recent tensorboard dir is right
    # if '_planner' not in tensorboard_dir:
    #     tensorboard_dir += '_planner'
    import datetime
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    tensorboard_dir = 'diffvg_compare/{}_planner'.format(run_name)
    writer = TensorBoard(tensorboard_dir)
    opt.writer = writer

    # n_stroke_test(opt)
    train_param2stroke(opt)
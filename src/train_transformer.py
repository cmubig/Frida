import datetime
import math
import os
import random
import sys
import numpy as np
import torch 
from torch import nn
from torchvision import models, transforms
import clip
from tqdm import tqdm
from torchvision.models import vgg16, resnet18
import torch.nn.functional as F
from transformers import ViTImageProcessor, BertTokenizer, VisionEncoderDecoderModel

from brush_stroke import BrushStroke
from options import Options
from paint_utils3 import format_img, load_img, show_img
from painting import Painting
from my_tensorboard import TensorBoard

device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
class StrokePredictor(nn.Module):
    def __init__(self, opt,
                 n_strokes=1,
                 stroke_latent_size=64):
        '''
            n_strokes (int) : number of strokes to predict with each forward pass
        '''
        super(StrokePredictor, self).__init__()
        self.stroke_latent_size = stroke_latent_size
        self.n_strokes = n_strokes
        self.opt = opt

        self.vit_model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "facebook/deit-tiny-patch16-224", "gaunernst/bert-tiny-uncased"
        #     "google/vit-base-patch16-224-in21k", "google-bert/bert-base-uncased"
        #     "facebook/deit-small-distilled-patch16-224", "gaunernst/bert-tiny-uncased"
        )
        # Replace first layer with new conv2d that can take 5 channels instead of 3 (add coords)
        # self.channel_reducer \
        #     = torch.nn.Conv2d(5,3, kernel_size=(5, 5), padding='same')

        # print(self.vit_model)

        image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.vit_model.config.decoder_start_token_id = tokenizer.cls_token_id
        self.vit_model.config.pad_token_id = tokenizer.pad_token_id

        self.vit_out_size = 128#768
        self.latent_head   = nn.Linear(self.vit_out_size, self.stroke_latent_size)
        self.position_head = nn.Linear(self.vit_out_size, 2)
        self.rotation_head = nn.Linear(self.vit_out_size, 1)
        # self.color_head    = nn.Linear(self.vit_out_size, 3)

        self.resize_normalize = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

        # size_x, size_y = 224, 224
        # idxs_x = torch.arange(size_x) / size_x
        # idxs_y = torch.arange(size_y) / size_y
        # x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # G x G
        # self.coords = torch.stack([x_coords, y_coords], dim=0).unsqueeze(0).to(device)

        # Labels is [batch_size, sequence_length]
        self.labels = torch.zeros((16, self.n_strokes), device=device, dtype=int)

    def forward(self, current_canvas, target_canvas, training=True):
        '''
            Given the current canvas and the target, predict the next n_strokes
            return:
                List[List[BrushStroke()]] : batch_size x n_strokes
        '''

        current_canvas = self.resize_normalize(current_canvas)
        target_canvas = self.resize_normalize(target_canvas)

        diff = target_canvas - current_canvas

        # if len(self.coords) < len(diff):
        #     self.coords = self.coords.repeat(len(diff),1,1,1) # So you don't have to repeatedly create this
        # coords = self.coords[:len(diff)]
        # coords = coords * diff.mean(dim=1).unsqueeze(1).repeat(1,2,1,1)
        # diff = torch.cat([diff, coords], dim=1)
        # diff = self.channel_reducer(diff)

        if len(self.labels) < len(diff):
            self.labels = torch.zeros((len(current_canvas), self.n_strokes), device=device, dtype=int)

        if training:
            # Labels is [batch_size, sequence_length]
            feats = self.vit_model(pixel_values=diff, output_hidden_states=True, labels=self.labels[:len(diff)])#.float()
        else:
            feats = self.vit_model(pixel_values=diff, output_hidden_states=True)#.float()
        
        # print('feats', feats)
            
        # print('encoder_last_hidden_state', feats.encoder_last_hidden_state.shape)
        # print('decoder_hidden_states', feats.decoder_hidden_states[-1].shape)
        # print('decoder_attentions', feats.decoder_attentions[-1].shape)

        feats = feats.decoder_hidden_states[-1]

        latents = self.latent_head(feats)#.float()
        # print('predicted latents size', latents.shape)
        position = self.position_head(feats)#.float()
        rotation = self.rotation_head(feats)#.float()
        # print('predicted rotation size', rotation.shape)
        # colors = self.color_head(feats)

        # Convert the output of the Transformer into BrushStroke Classes
        paintings = []
        brush_strokes_list = []
        for batch_ind in range(len(current_canvas)):
            predicted_brush_strokes = []
            for stroke_ind in range(self.n_strokes):
                latent = latents[batch_ind, stroke_ind, :]
                a =     rotation[batch_ind, stroke_ind, :]
                xt =    position[batch_ind, stroke_ind, :1]
                yt =    position[batch_ind, stroke_ind, -1:]
                bs = BrushStroke(self.opt, init_differentiably=True,
                                 ink=True,  latent=latent, a=a,
                                 xt=xt, yt=yt)
                predicted_brush_strokes.append(bs)
            # predicted_painting = Painting(self.opt,
            #         background_img=current_canvas[batch_ind:batch_ind+1], 
            #         brush_strokes=predicted_brush_strokes)
            # paintings.append(predicted_painting)
            brush_strokes_list.append(predicted_brush_strokes)
        return brush_strokes_list

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn 
    return pp

def get_random_painting(opt, n_strokes=random.randint(0,4), background_img=None):
    painting = Painting(opt, n_strokes=n_strokes, background_img=background_img)
    return painting

def get_random_brush_strokes(opt, n_strokes=random.randint(0,20)):
    painting = Painting(opt, n_strokes=n_strokes)
    return painting.brush_strokes

l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()

def stroke_distance(bs0, bs1):
    # Euclidian distance between starting points of two strokes
    return ((bs0.xt-bs1.xt)**2 + (bs0.yt-bs1.yt)**2)**0.5

from scipy.optimize import linear_sum_assignment
def match_brush_strokes(strokes0, strokes1):
    '''
        Re-order a list of strokes (strokes1) to have the strokes in an order
        such that comparing 1-to-1 to strokes0 minimizes differences in stroke position (x,y) 
        args:
            strokes0 List[BrushStroke()]
            strokes1 List[BrushStroke()]
        return:
            List[BrushStroke()] : the re-ordered strokes1
    '''
    # Create the cost matrix
    cost = np.empty((len(strokes0), len(strokes1)))
    for i in range(len(strokes0)):
        for j in range(len(strokes1)):
            cost[i,j] = stroke_distance(strokes0[i], strokes1[j])
    # print('cost\n', cost)
            
    # Perform linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    # print(row_ind, col_ind)
    
    # Re-order strokes1 to reduce costs
    reordered_strokes1 = [strokes1[i] for i in col_ind]
    # print(len(reordered_strokes1), len(strokes1), len(strokes0))

    # Confirm that the costs go down or are equal when comparing
    # cost_prev_order = np.sum(np.array([stroke_distance(strokes0[i], strokes1[min(i,len(strokes1)-1)]).cpu().detach().numpy() for i in range(len(strokes0))]))
    # cost_new_order = np.sum(np.array([stroke_distance(strokes0[i], reordered_strokes1[i]).cpu().detach().numpy() for i in range(len(strokes0))]))
    # if (cost_prev_order-cost_new_order) < 0:
    #     print(cost_prev_order, cost_new_order, cost_prev_order-cost_new_order, sep='\t')
    
    return reordered_strokes1

def wasserstein_distance(bs0, bs1):
    # Adapted from https://arxiv.org/pdf/2108.03798
    mu_u = torch.cat([bs0.xt, bs0.yt])
    # print('mu_u', mu_u)
    def sigma_wasserstein(theta):
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        return torch.Tensor([
            [cos_theta**2 + sin_theta**2, cos_theta*sin_theta],
            [cos_theta*sin_theta, sin_theta**2 + cos_theta**2]
        ])
    sigma_u = sigma_wasserstein(bs0.a)
    # print('sigma_u', sigma_u)
    mu_v = torch.cat([bs1.xt, bs1.yt])
    sigma_v = sigma_wasserstein(bs1.a)

    return l2_loss(mu_u, mu_v) \
        + torch.trace(sigma_u**2 + sigma_v**2 - (2*(sigma_u@(sigma_v**2)@sigma_u))**0.5)

def brush_stroke_parameter_loss_fcn(predicted_strokes, true_strokes):
    '''
        Calculate loss between brush strokes
        args:
            predicted_strokes List[List[BrushStroke()]]
            true_strokes List[List[BrushStroke()]]
    '''
    loss_x, loss_y, loss_rot, loss_latent, loss_wasserstein = 0,0,0,0,0

    for batch_ind in range(len(predicted_brush_strokes)):
        with torch.no_grad():
            true_strokes_reordered = match_brush_strokes(predicted_brush_strokes[batch_ind], true_strokes[batch_ind])
        for stroke_ind in range(len(predicted_brush_strokes[batch_ind])):
            pred_bs = predicted_strokes[batch_ind][stroke_ind]
            # true_bs = true_strokes_reordered[stroke_ind]
            true_bs = true_strokes_reordered[min(len(true_strokes_reordered)-1, stroke_ind)]

            loss_x += l1_loss(pred_bs.xt, true_bs.xt)
            # print(pred_bs.xt, true_bs.xt)
            loss_y += l1_loss(pred_bs.yt, true_bs.yt)
            loss_rot += l1_loss(pred_bs.a, true_bs.a)
            loss_latent += l1_loss(pred_bs.latent, true_bs.latent) #* 1e-3 ############################
            loss_wasserstein += wasserstein_distance(pred_bs, true_bs)

    n_batch, n_strokes = len(predicted_brush_strokes), len(predicted_brush_strokes[0])
    n = n_batch * n_strokes
    loss_x, loss_y, loss_rot, loss_latent, loss_wasserstein = loss_x/n, loss_y/n, loss_rot/n, loss_latent/n, loss_wasserstein/n

    loss = loss_x + loss_y + loss_rot + loss_latent + loss_wasserstein

    return loss, loss_x, loss_y, loss_rot, loss_latent, loss_wasserstein

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/sp_{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    save_dir = os.path.join('./stroke_predictor_models/', run_name)
    os.makedirs(save_dir, exist_ok=True)

    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render

    blank_canvas = load_img('../cofrida/blank_canvas.jpg',h=h_render, w=w_render).to(device)[:,:3]/255.

    stroke_predictor = StrokePredictor(opt, 
            n_strokes=opt.n_predicted_strokes)
            # n_strokes=2)
    if opt.continue_training is not None:
        stroke_predictor.load_state_dict(torch.load(opt.continue_training))

    stroke_predictor.to(device)
    for param in stroke_predictor.vit_model.parameters(): # These might not be necessary
        param.requires_grad = True
    for param in stroke_predictor.parameters():
        param.requires_grad = True

    print('# of parameters in stroke_predictor: ', get_n_params(stroke_predictor))
    print('# of parameters in latent_head: ', get_n_params(stroke_predictor.latent_head))
    print('# of parameters in vit_model: ', get_n_params(stroke_predictor.vit_model))
    print('# of parameters in encoder: ', get_n_params(stroke_predictor.vit_model.encoder))
    print('# of parameters in decoder: ', get_n_params(stroke_predictor.vit_model.decoder))
    
    optim = torch.optim.Adam(stroke_predictor.parameters(), lr=1e-4)

    pix_loss_fcn = torch.nn.L1Loss()
    # torch.autograd.set_detect_anomaly(True)
    batch_size = 24

    import torchvision.transforms.v2 as transforms
    target_img_aug = transforms.RandomPhotometricDistort(
        brightness=(0.75,1.25),
        contrast=(0.3,1.7),
        saturation=(0.3,1.7),
        hue=(-0.1,0.1),
        p=0.75
    )
    current_canvas_aug = transforms.RandomPhotometricDistort(
        brightness=(0.75,1.25),
        contrast=(0.3,1.7),
        saturation=(0.3,1.7),
        hue=(-0.1,0.1),
        p=0.75
    )
    blank_canvas_aug = transforms.Compose([
        current_canvas_aug,
        transforms.RandomResizedCrop((h_render,w_render), scale=(0.7, 1.0), ratio=(0.8,1.0))
    ])

    total_epochs = 100000
    for batch_ind in tqdm(range(total_epochs)):
        # print(stroke_predictor.latent_head.weight[0,:7]) # Check if weights are changing

        optim.param_groups[0]['lr'] *= 0.99995
        
        stroke_predictor.train()
        optim.zero_grad()

        current_canvases = []
        target_canvases = []
        true_brush_strokes = []
        predicted_brush_strokes = []
        predicted_next_canvases = []

        # Get the ground truth data
        for it in range(batch_size):
            # Get a current canvas
            with torch.no_grad():
                current_painting = get_random_painting(opt,
                        background_img=current_canvas_aug(blank_canvas)).to(device)
                current_canvas = current_painting(h_render, w_render, use_alpha=False)
                current_canvas = current_canvas_aug(current_canvas)
                current_canvases.append(current_canvas)

            # Generate a random brush stroke(s) to add. Render it to create the target canvas
            with torch.no_grad():
                true_brush_stroke = get_random_brush_strokes(opt, 
                        n_strokes=opt.n_predicted_strokes)
                        # n_strokes=random.randint(opt.n_predicted_strokes, 20)) # Variable number of target strokes

                # Render the strokes onto the current canvas
                target_painting = Painting(opt, background_img=current_canvases[it], 
                        brush_strokes=true_brush_stroke).to(device)
                target_canvas = target_painting(h_render, w_render, use_alpha=False)

                true_brush_strokes.append(true_brush_stroke)
                target_canvases.append(target_canvas)
            
        current_canvases = torch.cat(current_canvases, dim=0)
        target_canvases = torch.cat(target_canvases, dim=0)

        # Augment the target_canvases to reduce sim2real gap
        with torch.no_grad():
            # print(target_canvases.shape)
            target_canvases = target_img_aug(target_canvases)
            # print(target_canvases.shape)

        # Perform the prediction to estimate the added stroke(s)
        predicted_brush_strokes = stroke_predictor(current_canvases, target_canvases)

        # Render the predicted strokes
        for it in range(batch_size):
            # Render the strokes onto the current canvas
            predicted_painting = Painting(opt, background_img=current_canvases[it:it+1], 
                        brush_strokes=predicted_brush_strokes[it]).to(device)
            predicted_next_canvas = predicted_painting(h_render, w_render, use_alpha=False)
            predicted_next_canvases.append(predicted_next_canvas)
        predicted_next_canvases = torch.cat(predicted_next_canvases, dim=0)

        # Calculate losses. pix_loss in pixel space, and stroke_param_loss in stroke space
        pix_loss = pix_loss_fcn(predicted_next_canvases, target_canvases)
        stroke_param_loss, loss_x, loss_y, loss_rot, loss_latent, loss_wasserstein \
                = brush_stroke_parameter_loss_fcn(predicted_brush_strokes, true_brush_strokes)

        # Weight losses
        pix_loss_weight = (batch_ind / total_epochs)
        stroke_param_loss_weight = 1 - pix_loss_weight
        loss = pix_loss*pix_loss_weight + stroke_param_loss * stroke_param_loss_weight
        
        loss.backward()
        optim.step()

        # Log losses
        if batch_ind % 10 == 0:
            opt.writer.add_scalar('loss/pix_loss', pix_loss, batch_ind)
            opt.writer.add_scalar('loss/pix_loss_weight', pix_loss_weight, batch_ind)
            opt.writer.add_scalar('loss/stroke_param_loss', stroke_param_loss, batch_ind)
            opt.writer.add_scalar('loss/loss', loss, batch_ind)

            opt.writer.add_scalar('loss/loss_latent', loss_latent, batch_ind)
            opt.writer.add_scalar('loss/loss_x', loss_x, batch_ind)
            opt.writer.add_scalar('loss/loss_y', loss_y, batch_ind)
            opt.writer.add_scalar('loss/loss_rot', loss_rot, batch_ind)
            opt.writer.add_scalar('loss/loss_wasserstein', loss_wasserstein, batch_ind)

            opt.writer.add_scalar('loss/lr', optim.param_groups[0]['lr'], batch_ind)

        # Periodically save
        if batch_ind % 1000 == 0:
            torch.save(stroke_predictor.state_dict(), os.path.join(save_dir, 'stroke_predictor_weights.pth'))

        # Log images
        if batch_ind % 100 == 0:
            with torch.no_grad():
                # Log some images
                for log_ind in range(min(10, batch_size)):
                    # Log canvas with gt strokes, canvas with predicted strokes
                    t = target_canvases[log_ind:log_ind+1].clone()
                    t[:,:,:,-2:] = 0
                    log_img = torch.cat([t, predicted_next_canvases[log_ind:log_ind+1]], dim=3)
                    opt.writer.add_image('images/train{}'.format(str(log_ind)), 
                            format_img(log_img), batch_ind)
                    
                    # Log target_strokes-current_canvas and predicted_strokes-current_canvas
                    pred_diff_img = torch.abs(predicted_next_canvases[log_ind:log_ind+1] - current_canvases[log_ind:log_ind+1])
                    true_diff_img = torch.abs(target_canvases[log_ind:log_ind+1] - current_canvases[log_ind:log_ind+1])
                    
                    pred_diff_img_bool = pred_diff_img.mean(dim=1) > 0.3
                    true_diff_img_bool = true_diff_img.mean(dim=1) > 0.3
                    colored_img = torch.zeros(true_diff_img.shape).to(device)
                    colored_img[:,1][pred_diff_img_bool & true_diff_img_bool] = 1 # Green for true positives
                    colored_img[:,0][~pred_diff_img_bool & true_diff_img_bool] = 1 # Red for False negatives
                    colored_img[:,2][pred_diff_img_bool & ~true_diff_img_bool] = 1 # Blue for False positives
                    pred_diff_img[:,:,:,:2] = 1 # Draw a border
                    colored_img[:,:,:,:2] = 1 # Draw a border

                    log_img = torch.cat([true_diff_img, pred_diff_img, colored_img], dim=3)
                    opt.writer.add_image('images/train{}_diff'.format(str(log_ind)), 
                            format_img(log_img), batch_ind)
                    

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
from cofrida import get_instruct_pix2pix_model
from options import Options
from paint_utils3 import format_img, load_img, show_img
from painting import Painting
from my_tensorboard import TensorBoard

from losses.clip_loss import clip_conv_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from train_transformer import *

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
    
    optim = torch.optim.Adam(stroke_predictor.parameters(), lr=1e-5)

    pix_loss_fcn = torch.nn.L1Loss()

    def custom_pix_loss_fcn(canvas, goal_canvas):
        black_mask = (goal_canvas <= 0.2).float() # B x CANVAS_SIZE x CANVAS_SIZE
        white_mask = 1 - black_mask # B x CANVAS_SIZE x CANVAS_SIZE
        l2 = ((canvas - goal_canvas) ** 2) # B x CANVAS_SIZE x CANVAS_SIZE
        black_loss = (l2 * black_mask).mean(1).mean(1).mean(1) # B
        white_loss = (l2 * white_mask).mean(1).mean(1).mean(1) # B
        return (0.6 * black_loss + 0.4 * white_loss).mean() # 1

    # torch.autograd.set_detect_anomaly(True)
    batch_size = opt.sp_training_batch_size

    sd_interactive_pipeline = get_instruct_pix2pix_model(
        instruct_pix2pix_path=opt.cofrida_model, 
        device=device)

    import torchvision.transforms.v2 as transforms
    
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
    cofrida_target_img_aug = transforms.Compose([
        transforms.RandomPhotometricDistort(
            brightness=(0.75,1.05),
            contrast=(0.5,1.5),
            saturation=(0.5,1.5),
            hue=(-0.1,0.1),
            p=0.75
        ),
        transforms.RandomResizedCrop((h_render,w_render), scale=(0.7, 1.0), ratio=(0.8,1.0))
    ])

    total_epochs = 100000
    for batch_ind in tqdm(range(total_epochs)):
        # print(stroke_predictor.latent_head.weight[0,:7]) # Check if weights are changing

        # Generate a new bunch of target images every few iterations
        if batch_ind % 50 == 0:
            # Generate target images
            target_canvas_bank = []
            for it in range(batch_size*5):
                with torch.no_grad():
                    target_canvas_bank.append(sd_interactive_pipeline(
                        'A random drawing', 
                        blank_canvas, 
                        num_inference_steps=10, 
                        # generator=generator,
                        num_images_per_prompt=1,
                        output_type='pt',
                        # image_guidance_scale=2.5,#1.5 is default
                    ).images[0].unsqueeze(0))
            target_canvas_bank = torch.cat(target_canvas_bank, dim=0)

        optim.param_groups[0]['lr'] *= 0.99995
        
        stroke_predictor.train()
        optim.zero_grad()

        

        if batch_ind == 0:
            with torch.no_grad():
                current_canvases = blank_canvas.repeat((batch_size, 1,1,1)).to(device)
                current_canvases_og = current_canvases.detach().clone()

        # Get target canvases from the canvas bank
        with torch.no_grad():
            # print(target_canvases.shape)
            n_bank = len(target_canvas_bank)
            # Random sample from bank
            rand_ind = torch.randperm(n_bank)
            target_canvases = target_canvas_bank[rand_ind[:batch_size]]
            # Augment the target_canvases to reduce sim2real gap
            target_canvases = cofrida_target_img_aug(target_canvases)
            # print(target_canvases.shape)

        

        if opt.predict_many_then_optimize:
            optim.zero_grad()
            current_canvases = current_canvases_og.clone()
            custom_pix_loss_tot, pix_loss_tot, clip_loss_tot, loss_tot = 0,0,0,0

            for pred_it in range(opt.num_prediction_rounds): # num times to predict a batch of strokes
                predicted_brush_strokes = []
                predicted_next_canvases = []

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

                current_canvases = predicted_next_canvases#.detach()

            # Calculate losses. pix_loss in pixel space, and stroke_param_loss in stroke space
            pix_loss = pix_loss_fcn(predicted_next_canvases, target_canvases)
            custom_pix_loss = custom_pix_loss_fcn(predicted_next_canvases, target_canvases)
            clip_loss = clip_conv_loss(target_canvases, predicted_next_canvases[:,:3]) 
            loss = custom_pix_loss + clip_loss
            
            custom_pix_loss_tot += custom_pix_loss.item()
            pix_loss_tot += pix_loss.item()
            clip_loss_tot += clip_loss.item()
            loss_tot += loss.item()

            loss.backward()
            optim.step()
        else:
            # current_canvases = current_canvases_og.clone()
            # custom_pix_loss_tot, pix_loss_tot, clip_loss_tot, loss_tot = 0,0,0,0
            # for pred_it in range(opt.num_prediction_rounds): # num times to predict a batch of strokes
            #     predicted_brush_strokes = []
            #     predicted_next_canvases = []

            #     optim.zero_grad()

            #     # Perform the prediction to estimate the added stroke(s)
            #     predicted_brush_strokes = stroke_predictor(current_canvases, target_canvases)

            #     # Render the predicted strokes
            #     for it in range(batch_size):
            #         # Render the strokes onto the current canvas
            #         predicted_painting = Painting(opt, background_img=current_canvases[it:it+1], 
            #                     brush_strokes=predicted_brush_strokes[it]).to(device)
            #         predicted_next_canvas = predicted_painting(h_render, w_render, use_alpha=False)
            #         predicted_next_canvases.append(predicted_next_canvas)
            #     predicted_next_canvases = torch.cat(predicted_next_canvases, dim=0)

            #     # Calculate losses. pix_loss in pixel space, and stroke_param_loss in stroke space
            #     pix_loss = pix_loss_fcn(predicted_next_canvases, target_canvases)
                
            #     custom_pix_loss = custom_pix_loss_fcn(predicted_next_canvases, target_canvases)

            #     clip_loss = clip_conv_loss(target_canvases, predicted_next_canvases[:,:3]) 

            #     loss = custom_pix_loss + clip_loss
                
            #     custom_pix_loss_tot += custom_pix_loss.item()
            #     pix_loss_tot += pix_loss.item()
            #     clip_loss_tot += clip_loss.item()
            #     loss_tot += loss.item()

            #     loss.backward()
            #     optim.step()

            #     current_canvases = predicted_next_canvases.detach()
            current_canvases = current_canvases_og.clone()
            custom_pix_loss_tot, pix_loss_tot, clip_loss_tot, loss_tot = 0,0,0,0
            for pred_it in range(opt.num_prediction_rounds): # num times to predict a batch of strokes
                predicted_brush_strokes = []
                predicted_next_canvases = []
                optim.zero_grad()

                # Perform the prediction to estimate the added stroke(s)
                predicted_brush_strokes = stroke_predictor(current_canvases, target_canvases)

                # Render the predicted strokes
                for it in range(batch_size):
                    

                    # Render the strokes onto the current canvas
                    predicted_painting = Painting(opt, background_img=current_canvases[it:it+1], 
                                brush_strokes=predicted_brush_strokes[it]).to(device)
                    predicted_next_canvas = predicted_painting(h_render, w_render, use_alpha=False)
                    predicted_next_canvases.append(predicted_next_canvas.detach())


                    # Calculate losses. pix_loss in pixel space, and stroke_param_loss in stroke space
                    target_canvas = target_canvases[it:it+1]
                    pix_loss = pix_loss_fcn(predicted_next_canvas, target_canvas)
                    
                    custom_pix_loss = custom_pix_loss_fcn(predicted_next_canvas, target_canvas)

                    clip_loss = clip_conv_loss(target_canvas, predicted_next_canvas[:,:3]) 

                    loss = custom_pix_loss*0.25 + clip_loss*0.75
                    
                    custom_pix_loss_tot += custom_pix_loss.item()
                    pix_loss_tot += pix_loss.item()
                    clip_loss_tot += clip_loss.item()
                    loss_tot += loss.item()

                    loss.backward(retain_graph=True)
                optim.step()

                predicted_next_canvases = torch.cat(predicted_next_canvases, dim=0)
                current_canvases = predicted_next_canvases.detach()
            


        # Log losses
        if batch_ind % 10 == 0:
            opt.writer.add_scalar('loss/pix_loss', pix_loss_tot, batch_ind)
            opt.writer.add_scalar('loss/custom_pix_loss', custom_pix_loss_tot, batch_ind)
            opt.writer.add_scalar('loss/clip_loss', clip_loss_tot, batch_ind)
            opt.writer.add_scalar('loss/loss', loss_tot, batch_ind)

            opt.writer.add_scalar('loss/lr', optim.param_groups[0]['lr'], batch_ind)

        # Periodically save
        if batch_ind % 1000 == 0:
            torch.save(stroke_predictor.state_dict(), os.path.join(save_dir, 'stroke_predictor_weights.pth'))

        # Log images
        if batch_ind % 50 == 0:
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
                    pred_diff_img = torch.abs(predicted_next_canvases[log_ind:log_ind+1] - current_canvases_og[log_ind:log_ind+1])
                    true_diff_img = torch.abs(target_canvases[log_ind:log_ind+1] - current_canvases_og[log_ind:log_ind+1])
                    
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
                    

import datetime
import random
import sys
import numpy as np
import torch 
from torch import nn
from torchvision import models, transforms
import clip
from tqdm import tqdm
from torchvision.models import vgg16, resnet18

from brush_stroke import BrushStroke
from options import Options
from paint_utils3 import format_img, show_img
from painting import Painting
from my_tensorboard import TensorBoard

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class StrokePredictor(nn.Module):
    def __init__(self, opt,
                 clip_model_name='ViT-B/32', 
                 n_strokes=1,
                 stroke_latent_size=64):
        '''
            n_strokes (int) : number of strokes to predict with each forward pass
        '''
        super(StrokePredictor, self).__init__()
        self.stroke_latent_size = stroke_latent_size
        self.n_strokes = n_strokes
        self.opt = opt

        # self.vit_model, clip_preprocess \
        #     = clip.load(clip_model_name, device, jit=False)
        # self.vit_model = vgg16()
        self.vit_model = resnet18(weights='DEFAULT')

        self.vit_out_size = 1000#512 # self.vit_model.ln_final.shape 

        # Define output layers for each of the brush stroke variables
        self.latent_head   = nn.Linear(self.vit_out_size, self.stroke_latent_size * n_strokes)
        self.position_head = nn.Linear(self.vit_out_size, 2 * n_strokes)
        self.rotation_head = nn.Linear(self.vit_out_size, 1 * n_strokes)
        self.color_head    = nn.Linear(self.vit_out_size, 3 * n_strokes)

        self.resize_normalize = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def forward(self, current_canvas, target_canvas):

        current_canvas = self.resize_normalize(current_canvas)
        target_canvas = self.resize_normalize(target_canvas)

        diff = target_canvas - current_canvas

        # feats = self.vit_model.encode_image(diff)#.float()
        feats = self.vit_model(diff)#.float()
        
        latents = self.latent_head(feats)#.float()
        # print('predicted latents size', latents.shape)
        position = self.position_head(feats)#.float()
        rotation = self.rotation_head(feats)#.float()
        # print('predicted rotation size', rotation.shape)
        # colors = self.color_head(feats)

        paintings = []
        brush_strokes_list = []
        for batch_ind in range(len(current_canvas)):
            predicted_brush_strokes = []
            for stroke_ind in range(self.n_strokes):
                latent = latents[batch_ind, self.stroke_latent_size*stroke_ind:self.stroke_latent_size*(stroke_ind+1)]
                a =     rotation[batch_ind, stroke_ind:stroke_ind+1]
                xt =    position[batch_ind,stroke_ind*2:stroke_ind*2+1]
                yt =    position[batch_ind,stroke_ind*2+1:stroke_ind*2+2]
                bs = BrushStroke(self.opt, init_differentiably=True,
                                 ink=True,  latent=latent, a=a,
                                 xt=xt, yt=yt)
                predicted_brush_strokes.append(bs)
            # predicted_painting = Painting(self.opt,
            #         background_img=current_canvas[batch_ind:batch_ind+1], 
            #         brush_strokes=predicted_brush_strokes)
            # paintings.append(predicted_painting)
            brush_strokes_list.append(predicted_brush_strokes)
        return brush_strokes_list, latents

def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn 
    return pp

def get_random_painting(opt, n_strokes=random.randint(0,20)):
    painting = Painting(opt, n_strokes=n_strokes)
    return painting

def get_random_brush_strokes(opt, n_strokes=random.randint(0,20)):
    painting = Painting(opt, n_strokes=n_strokes)
    return painting.brush_strokes

l1_loss = torch.nn.L1Loss()


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
    
    # Re-order strokes1 to reduce costs
    reordered_strokes1 = [strokes1[i] for i in col_ind]

    # Confirm that the costs go down or are equal when comparing
    # cost_prev_order = np.sum(np.array([stroke_distance(strokes0[i], strokes1[i]).cpu().detach().numpy() for i in range(len(strokes0))]))
    # cost_new_order = np.sum(np.array([stroke_distance(strokes0[i], reordered_strokes1[i]).cpu().detach().numpy() for i in range(len(strokes0))]))
    # print(cost_prev_order, cost_new_order, cost_prev_order-cost_new_order, sep='\t')
    
    return reordered_strokes1


def brush_stroke_parameter_loss_fcn(predicted_strokes, true_strokes):
    '''
        Calculate loss between brush strokes
        TODO: Add hungarian matching loss 
        args:
            predicted_strokes List[List[BrushStroke()]]
            true_strokes List[List[BrushStroke()]]
    '''
    loss_x, loss_y, loss_rot, loss_latent = 0,0,0,0

    for batch_ind in range(len(predicted_brush_strokes)):
        true_strokes_reordered = match_brush_strokes(predicted_brush_strokes[batch_ind], true_strokes[batch_ind])
        for stroke_ind in range(len(predicted_brush_strokes[batch_ind])):
            pred_bs = predicted_strokes[batch_ind][stroke_ind]
            true_bs = true_strokes_reordered[stroke_ind]

            loss_x += l1_loss(pred_bs.xt, true_bs.xt)
            # print(pred_bs.xt, true_bs.xt)
            loss_y += l1_loss(pred_bs.yt, true_bs.yt)
            loss_rot += l1_loss(pred_bs.a, true_bs.a)
            loss_latent += l1_loss(pred_bs.latent, true_bs.latent) * 1e-2 ############################
    
    loss = loss_x + loss_y + loss_rot + loss_latent 
    loss /= len(predicted_brush_strokes)

    n = len(predicted_brush_strokes)
    loss_x, loss_y, loss_rot, loss_latent = loss_x/n, loss_y/n, loss_rot/n, loss_latent/n
    return loss, loss_x, loss_y, loss_rot, loss_latent

if __name__ == '__main__':
    opt = Options()
    opt.gather_options()

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/sp_{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render

    stroke_predictor = StrokePredictor(opt, 
            n_strokes=opt.n_predicted_strokes)
    stroke_predictor.to(device)
    for param in stroke_predictor.vit_model.parameters(): # These might not be necessary
        param.requires_grad = True
    for param in stroke_predictor.parameters():
        param.requires_grad = True

    print('# of parameters in stroke_predictor: ', get_n_params(stroke_predictor))
    
    optim = torch.optim.Adam(stroke_predictor.parameters(), lr=1e-2)

    pix_loss_fcn = torch.nn.L1Loss()
    # torch.autograd.set_detect_anomaly(True)
    batch_size = 32

    for batch_ind in tqdm(range(200000)):
        # print(stroke_predictor.latent_head.weight[0,:7]) # Check if weights are changing
        
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
                current_painting = get_random_painting(opt).to(device)
                current_canvas = current_painting(h_render, w_render, use_alpha=False)
                current_canvases.append(current_canvas)

            # Generate a random brush stroke(s) to add. Render it to create the target canvas
            with torch.no_grad():
                true_brush_stroke = get_random_brush_strokes(opt, n_strokes=opt.n_predicted_strokes)

                # Render the strokes onto the current canvas
                target_painting = Painting(opt, background_img=current_canvases[it], 
                        brush_strokes=true_brush_stroke).to(device)
                target_canvas = target_painting(h_render, w_render, use_alpha=False)

                true_brush_strokes.append(true_brush_stroke)
                target_canvases.append(target_canvas)
            
        current_canvases = torch.cat(current_canvases, dim=0)
        target_canvases = torch.cat(target_canvases, dim=0)

        # Perform the prediction to estimate the added stroke(s)
        predicted_brush_strokes, lat = stroke_predictor(current_canvases, target_canvases)

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
        stroke_param_loss, loss_x, loss_y, loss_rot, loss_latent \
                = brush_stroke_parameter_loss_fcn(predicted_brush_strokes, true_brush_strokes)

        # Weight losses
        loss = pix_loss*1e-1 + stroke_param_loss
        
        loss.backward()
        optim.step()

        # Log losses
        if batch_ind % 10 == 0:
            opt.writer.add_scalar('loss/pix_loss', pix_loss, batch_ind)
            opt.writer.add_scalar('loss/stroke_param_loss', stroke_param_loss, batch_ind)
            opt.writer.add_scalar('loss/loss', loss, batch_ind)

            opt.writer.add_scalar('loss/loss_latent', loss_latent, batch_ind)
            opt.writer.add_scalar('loss/loss_x', loss_x, batch_ind)
            opt.writer.add_scalar('loss/loss_y', loss_y, batch_ind)
            opt.writer.add_scalar('loss/loss_rot', loss_rot, batch_ind)

        # Log images
        if batch_ind % 100 == 0:
            with torch.no_grad():
                # Log some images
                for log_ind in range(min(10, batch_size)):
                    t = target_canvases[log_ind:log_ind+1]
                    t[:,:,:,-5:] = 0
                    log_img = torch.cat([t, predicted_next_canvases[log_ind:log_ind+1]], dim=3)
                    opt.writer.add_image('images/train{}'.format(str(log_ind)), 
                            format_img(log_img), batch_ind)
                    t = target_canvases[log_ind:log_ind+1]
                    t[:,:,:,-5:] = 0
                    pred_diff_img = torch.abs(predicted_next_canvases[log_ind:log_ind+1] - current_canvases[log_ind:log_ind+1])
                    true_diff_img = torch.abs(target_canvases[log_ind:log_ind+1] - current_canvases[log_ind:log_ind+1])
                    log_img = torch.cat([true_diff_img, pred_diff_img], dim=3)
                    opt.writer.add_image('images/train{}_diff'.format(str(log_ind)), 
                            format_img(log_img), batch_ind)
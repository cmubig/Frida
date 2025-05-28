import os
import clip
import torch
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from matplotlib import pyplot as plt

from painting import PaintingBatch
from brush_stroke import BrushStrokeBatch
from stroke_generator.model import StrokePredictor

class OfflineMultiStrokeGeneratorTrainer():
    def __init__(self, opt, model: StrokePredictor, device, dataloader, save_folder='', run_name='run'):
        self.opt = opt
        self.model = model
        self.device = device
        self.dataloader = dataloader

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=100)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=10, T_mult=2)

        # Canvas params
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

        # Save params
        self.save_folder = save_folder
        self.run_name = run_name
    
    def train(self, epochs):
        epoch_losses = []
        tokenized_text = clip.tokenize(["A painting"]*self.dataloader.batch_size).detach().to(self.device)
        tokenized_text.requires_grad = False
        self.lr_scheduler.T_0 = max(epochs//32,1)
        best_loss = 1e10

        for e in range(epochs):
            losses = []
            for i, data in tqdm(enumerate(self.dataloader), desc="Training Epoch...", unit="Batch", total=len(self.dataloader)):
                # Shape batch_size, n_strokes+1, 3, h_render, w_render
                canvases = data[0].to(self.device)
                n_strokes = canvases.shape[1]-1
                strokes_gt = data[1].detach().to(self.device)

                canvas_current = canvases[:,:-1]
                canvas_target = canvases[:,-1]
                # remaining strokes should be N,N-1...1 for size [BatchSize, N, 1]
                remaining_strokes = torch.arange(n_strokes, 0, -1)
                remaining_strokes = remaining_strokes.unsqueeze(0).unsqueeze(-1).repeat(canvas_current.shape[0],1,1).to(self.device).to(torch.float)                 

                # Set up masks. Masked items are stored as all -1s
                invalid_slots = (canvas_current == -1).all(dim=-1).all(dim=-1).all(dim=-1)
                mask = torch.where(invalid_slots, torch.zeros_like(invalid_slots), torch.ones_like(invalid_slots)).to(torch.float)

                # repeat canvas_target for all canvas_current
                canvas_target = canvas_target.unsqueeze(1).repeat(1,n_strokes,1,1,1)
                color_palette = data[2].unsqueeze(1).repeat(1,n_strokes,1,1).to(self.device)
                color_gt_onehot = (strokes_gt[...,-3:].unsqueeze(-2).repeat(1,1,12,1)==color_palette).all(dim=-1).to(torch.float)
                tok_text = tokenized_text[0:canvas_current.shape[0]].unsqueeze(1).repeat(1,n_strokes,1)

                # Forward pass
                lzbaxy_mean, lzbaxy_log_std, rgb_class = self.model(
                    current_canvas=canvas_current,
                    target_img=canvas_target,
                    target_tokenized_txt=tok_text,
                    remaining_strokes=remaining_strokes,
                    color_palette=color_palette,
                    mask=mask
                )
                lzbaxy_std = torch.exp(lzbaxy_log_std)

                self.optim.zero_grad()
                loss = self.dist_loss_fn(lzbaxy_mean, lzbaxy_std, color_gt_onehot, rgb_class, strokes_gt, mask) + \
                        self.canv_loss_fn(canvas_current, canvas_target, mask)
                # loss = self.loss_fn(strokes_pred*stroke_mask, strokes_gt*stroke_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optim.step()

                losses.append(loss.item())
            
            if self.lr_scheduler is not None: self.lr_scheduler.step()
            epoch_losses.append(torch.tensor(losses).mean().item())
            print(f"Epoch: {e+1}/{epochs} | | Loss: {epoch_losses[-1]:0.4f}")

            # Paint ground truth brush stroke and save image after every epoch
            with torch.no_grad():
                # Save stroke examples
                if e%20==0 or e==epochs-1: # (epochs//20)
                    strokes = self.model.sample(
                        current_canvas=canvas_current,
                        target_img=canvas_target,
                        target_tokenized_txt=tok_text,
                        remaining_strokes=remaining_strokes,
                        color_palette=color_palette,
                        mask=mask
                    )
                    for i in range(strokes.shape[0]):
                        s = strokes[i].unsqueeze(0)
                        n = (mask[i]==1).sum()
                        s_p = s[:,-n:]
                        c_c = canvas_current[i,-n:]
                        c_t = canvas_target[i,0]
                        self.paint_and_save(s_p, c_c, c_t, f'_{self.run_name}_{e}_{i}')
                # Plot losses
                plt.close()
                plt.plot(epoch_losses)
                # plt.yscale('log')
                plt.savefig(os.path.join(self.save_folder,f"losses_{self.run_name}.png"))
                # plt.yscale('linear')
                # Save model
                if epoch_losses[-1] < best_loss:
                    best_loss = epoch_losses[-1]
                    torch.save(self.model.state_dict(), os.path.join(self.save_folder,f"stroke_generator_state_dict.pth"))

        torch.save(torch.tensor(losses), os.path.join(self.save_folder,f"losses_{self.run_name}.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.save_folder,f"stroke_generator_state_dict_{self.run_name}.pth"))
        return self.model

    def dist_loss_fn(self, lzbaxy_mu, lzbaxy_std, rgb_onehot, rgb_logits, y_gt, mask):
        # Split up the lzbaxy_mu into individual components
        lzb_mu, lzb_std = lzbaxy_mu[:,:,0:3], lzbaxy_std[:,:,0:3]
        lzb_gt = y_gt[:,:,0:3]
        a_mu, a_std = lzbaxy_mu[:,:,3:5], lzbaxy_std[:,:,3:5]
        a_gt = y_gt[:,:,3]
        xy_mu, xy_std = lzbaxy_mu[:,:,5:], lzbaxy_std[:,:,5:]
        xy_gt = y_gt[:,:,4:6]

        # Length, Z, Bend loss
        lzb_dist = torch.distributions.Normal(lzb_mu, lzb_std)
        lzb_loss = -lzb_dist.log_prob(lzb_gt)

        # Breaks down angle into X and Y components to avoid discontinuities
        a_xy_gt = torch.cat([torch.sin(a_gt).unsqueeze(-1), torch.cos(a_gt).unsqueeze(-1)],dim=-1)
        a_dist = torch.distributions.Normal(a_mu, a_std)
        a_loss = -a_dist.log_prob(a_xy_gt)

        # X and Y loss
        xy_dist = torch.distributions.Normal(xy_mu, xy_std)
        xy_loss = -xy_dist.log_prob(xy_gt)

        # RGB loss (Categorical b/c of palette)
        rgb_dist = torch.distributions.Categorical(logits=rgb_logits)  # Categorical distribution
        rgb_loss = -rgb_dist.log_prob(rgb_onehot.argmax(dim=-1)).unsqueeze(-1)

        # Mask, Sum, and Normalize loss
        total_loss = torch.cat([lzb_loss, a_loss, xy_loss, rgb_loss], dim=-1)
        total_loss = (total_loss * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-6)

        return total_loss

    def canv_loss_fn(self, canvas_current, canvas_target, mask):
        canv_mse = ((canvas_current - canvas_target)**2).mean(dim=-1).mean(dim=-1).mean(dim=-1)
        # Mask out the invalid pixels
        canv_mse = (canv_mse * mask).sum() / (mask.sum() + 1e-6)
        return canv_mse
    
    def paint_and_save(self, stroke_tensors, canvas_current, canvas_target, name=''):
        # Model outputs [l, z, b, a, x, y, r, g, b]
        stroke_tensors = stroke_tensors.unsqueeze(-1)
        strokes = []
        for i in range(stroke_tensors.shape[1]):
            generated_stroke = BrushStrokeBatch(self.opt,
                                        stroke_length=stroke_tensors[:,i,0],
                                        stroke_z=stroke_tensors[:,i,1],
                                        stroke_bend=stroke_tensors[:,i,2],
                                        stroke_alpha=torch.zeros(stroke_tensors.shape[0],1).to(self.device),
                                        color=stroke_tensors[:,i,6:].squeeze(-1),
                                        a=stroke_tensors[:,i,3],
                                        xt=stroke_tensors[:,i,4],
                                        yt=stroke_tensors[:,i,5],
                                        init_differentiably=True,
                                        ink=None)
            strokes.append(generated_stroke)
        generated_painting = PaintingBatch(self.opt, background_img=canvas_current).to(self.device)
        generated_canvas = generated_painting(strokes, self.h_render, self.w_render, use_alpha=False, return_alphas=False)
        
        joint_canvas = torch.cat([canvas_target, torch.zeros((3,canvas_target.shape[1],3)).to(self.device), generated_canvas[0]], dim=-1)
        
        self.save_canvas_as_img(joint_canvas, path=self.save_folder, suffix=name)
    
    def save_canvas_as_img(self, canvas, path='outputs', suffix=''):
        plt.close()
        plt.imshow(canvas.detach().cpu().permute(1,2,0).numpy())
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except:
                raise Exception(f"Path provided for saving canvas does not exist and couldn't be created.\n\t{path}")
        plt.savefig(os.path.join(path,f"canvas{suffix}.png"))

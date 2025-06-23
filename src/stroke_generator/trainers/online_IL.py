import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import clip
import torch
import torch.nn.functional as F

from painting import PaintingBatch
from brush_stroke import BrushStrokeBatch
from stroke_generator.model import StrokePredictor
from stroke_generator.SAC.random_expert import RandomExpert, GridExpert
from stroke_generator.encoders.stroke_encoder import StrokeNetEncoder


class OnlineMultiStrokeGeneratorTrainer():
    def __init__(self, opt, model: StrokePredictor, device, batch_size=16, max_strokes=10, save_folder='', run_name='run'):
        self.opt = opt
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.max_strokes = max_strokes

        self.optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=5e-4
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=10, T_mult=2)
        self.stroke_encoder = StrokeNetEncoder().to(device)

        # Canvas params
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

        # Save params
        self.save_folder = save_folder
        self.run_name = run_name

        self.horizon = 10
        self.generate_new_target = True
        self.random_expert = GridExpert(opt, device)
        self.cur_max_strokes = max_strokes
        self.starting_canvas = torch.ones(self.batch_size, 3, self.h_render, self.w_render).to(self.device)
        self.tokenized_text = clip.tokenize(["A splash of colors on a white background"]*self.batch_size).detach().to(self.device)
        self.gt_strokes = None
        self.gt_canvases = None

        # Initializations
        self.state = None
        self.prev_init_state = None
        self.model.save_hx = False
        self.hx = None
        self.noise_level = 0.001

    def env_reset(self):
        # Reset environment to blank canvas and random color palette
        if self.generate_new_target:
            # Create new canvas and rollout trajectory
            starting_canvas = self.starting_canvas.clone()
            color_palette = torch.rand(self.batch_size, 12, 3).to(self.device)
            remaining_strokes = torch.randint(1, self.max_strokes+1, (self.batch_size,)).unsqueeze(-1).to(self.device).to(torch.float)
            gt_strokes, gt_canvases = self.random_expert.rollout_trajectory(remaining_strokes.clone(), starting_canvas.clone(), color_palette.clone())
            
            # extract state information
            self.gt_strokes = gt_strokes.detach().clone()
            self.gt_canvases = gt_canvases[:,:-1].detach().clone()
            self.cur_max_strokes = gt_strokes.shape[1]
            target_img = gt_canvases[torch.arange(self.batch_size), remaining_strokes[:,0].to(torch.long)].clone()
            
            # Set up other state information
            tokenized_text = self.tokenized_text.clone()
            mask = torch.ones(self.batch_size, 1).to(self.device)

            # Expand each to [batch_size, horizon, ...]
            curr_canv = self.gt_canvases[:,:self.horizon].detach().clone()
            targ_img = target_img.unsqueeze(1).repeat(1,curr_canv.shape[1],1,1,1)
            tokenized_text = tokenized_text.unsqueeze(1).repeat(1,curr_canv.shape[1],1)
            color_palette = color_palette.unsqueeze(1).repeat(1,curr_canv.shape[1],1,1)
            
            # remaining strokes from [batch_size, n] to [batch_size, [n, n-1, n-2...]]
            remaining_strokes = remaining_strokes.unsqueeze(1).repeat(1,self.gt_canvases.shape[1],1) - torch.arange(self.gt_canvases.shape[1]).unsqueeze(0).unsqueeze(-1).to(self.device)
            # Set up GT mask anywhere there are no strokes left
            self.gt_mask = torch.where(remaining_strokes <= 0, torch.zeros(self.gt_canvases.shape[1],1).to(self.device), torch.ones(self.gt_canvases.shape[1],1).to(self.device)).to(torch.float)
            # Set up local mask and strokes remaining
            remaining_strokes = remaining_strokes[:,:curr_canv.shape[1]]
            mask = self.gt_mask[:,:curr_canv.shape[1]].clone()

            self.state = {
                "current_canvas": curr_canv,
                "target_img": targ_img,
                "target_tokenized_txt": tokenized_text,
                "remaining_strokes": remaining_strokes,
                "color_palette": color_palette,
                "mask": mask,
            }
            self.prev_init_state = {k:v.clone() for k,v in self.state.items()}
            self.gt_strokes = self.gt_strokes.cpu()
            self.gt_canvases = self.gt_canvases.cpu()
            self.gt_mask = self.gt_mask.cpu()
        else:
            self.state = {k:v.clone() for k,v in self.prev_init_state.items()}

        self.hx = None
    
    def train(self, epochs):
        epoch_losses = []
        self.lr_scheduler.T_0 = max(epochs//32,1)
        best_loss = 1e10

        for e in range(epochs):
            losses = []

            with torch.no_grad():
                self.env_reset()
            self.model.hx = None
            for i in tqdm(range(self.cur_max_strokes), desc="Training Epoch...", unit="Stroke", total=self.cur_max_strokes):
                # Unpack state information
                current_canvas = self.state['current_canvas']
                target_img = self.state['target_img']
                target_tokenized_txt = self.state['target_tokenized_txt']
                color_palette = self.state['color_palette']
                remaining_strokes = self.state['remaining_strokes']
                mask = self.state['mask']

                # Add noise to the current canvas and target image
                noisy_current_canvas = current_canvas + torch.randn_like(current_canvas) * self.noise_level
                noisy_target_image = target_img + torch.randn_like(target_img) * self.noise_level

                # Get distributions from model forward pass
                lzbaxy_mean, lzbaxy_log_std, rgb_logits = self.model(
                    current_canvas=noisy_current_canvas,
                    target_img=noisy_target_image,
                    target_tokenized_txt=target_tokenized_txt,
                    remaining_strokes=remaining_strokes,
                    color_palette=color_palette,
                    mask=mask
                )

                # Get loss
                gt_strokes_copy = self.gt_strokes[:,i:].clone().to(self.device)
                gt_mask_copy = self.gt_mask[:,i:].clone().to(self.device)
                matching_loss = self.nonrepeating_softmatch_loss(
                    lzbaxy_mean, 
                    lzbaxy_log_std.exp(), 
                    rgb_logits, 
                    gt_strokes_copy,
                    gt_mask_copy
                )
                set_loss = self.set_loss(
                    lzbaxy_mean,
                    lzbaxy_log_std.exp(),
                    rgb_logits,
                    gt_strokes_copy,
                    gt_mask_copy
                )
                loss = 0.01*matching_loss + 100*set_loss

                # Backward pass
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optim.step()
                losses.append(loss.item())

                # Update state for next iter
                next_state = {k:v.clone() for k,v in self.state.items()}
                next_state['remaining_strokes'] -= 1
                next_state['mask'] = torch.where(self.state['remaining_strokes'] <= 0, torch.zeros_like(self.state['mask']), torch.ones_like(self.state['mask'])).to(torch.float)
                
                # Shift window
                curr_horizon = min(self.horizon, self.cur_max_strokes - (i+1))
                next_state['current_canvas'] = self.gt_canvases[:,i+1:i+1+curr_horizon].clone().to(self.device)
                next_state['target_img'] = self.state['target_img'][:,:curr_horizon].clone()
                next_state['target_tokenized_txt'] = self.state['target_tokenized_txt'][:,:curr_horizon].clone()
                next_state['color_palette'] = self.state['color_palette'][:,:curr_horizon].clone()
                next_state['remaining_strokes'] = next_state['remaining_strokes'][:,:curr_horizon]
                next_state['mask'] = next_state['mask'][:,:curr_horizon]

                self.state = next_state
                # self.model.hx = tuple(h.detach() for h in self.model.hx)

            if self.lr_scheduler is not None: self.lr_scheduler.step()
            epoch_losses.append(torch.tensor(losses).mean().item())
            print(f"Epoch: {e+1}/{epochs}  |  Loss: {epoch_losses[-1]:0.5f}")

            if e%(epochs//5)==0 or e==epochs-1:
                with torch.inference_mode():
                    # Rollout a single trajectory and save
                    self.paint_example_canvas(f"canvas_{e}")
            # Plot losses
            plt.close()
            plt.plot(epoch_losses)
            plt.yscale('log')
            plt.savefig(os.path.join(self.save_folder,f"losses_{self.run_name}.png"))
            plt.yscale('linear')
            # Save model
            if epoch_losses[-1] < best_loss:
                best_loss = epoch_losses[-1]
                torch.save(self.model.state_dict(), os.path.join(self.save_folder,f"stroke_generator_state_dict.pth"))

        torch.save(torch.tensor(losses), os.path.join(self.save_folder,f"losses_{self.run_name}.pth"))
        return self.model
    
    def breakdown_stroke(self, strokes, color_classes=12):
        B, H = strokes.shape[:2]

        # Process expert actions into LZBAxAyXY tensor and RGB onehot
        angle = strokes[:,:,3]
        cartesian = torch.cat([torch.sin(angle).unsqueeze(-1), torch.cos(angle).unsqueeze(-1)],dim=-1)
        lzbaxy = torch.cat([strokes[:,:,:3], cartesian, strokes[:,:,4:6]], dim=-1)
        
        # Reparametrize into [-1,1]
        lzbaxy = (lzbaxy - self.model.bias) / self.model.scale * 2.0
        
        # RGB One hot encoding
        rgb = strokes[:,:,-3:]
        rgb_onehot = (rgb.unsqueeze(-2).repeat(1,1,color_classes,1)==self.state['color_palette'][:,0].unsqueeze(-3).repeat(1,H,1,1)).all(dim=-1).to(torch.float)
        rgb_class = rgb_onehot.argmax(dim=-1)

        return lzbaxy, rgb_onehot, rgb_class

    def nonrepeating_softmatch_loss(self, lzbaxy_mean, lzbaxy_std, rgb_logits, expert_actions, mask):
        B, H, C = rgb_logits.shape[:3]
        N = expert_actions.shape[1]

        # Process expert actions into LZBAxAyXY tensor and RGB onehot
        expert_lzbaxy, _, expert_rgb_class = self.breakdown_stroke(expert_actions)

        total_loss, count = 0, 0
        used_mask = mask.clone()
        for t in range(H):
            # Expand predictions for [t] to all remaining strokes
            # Expand predicted distributions at time t to match target count
            pred_lzbaxy_mean = lzbaxy_mean[:, t].unsqueeze(1).expand(-1, N - t, -1)  # [B, N-t-1, 7]
            pred_lzbaxy_std = lzbaxy_std[:, t].unsqueeze(1).expand(-1, N - t, -1)
            pred_rgb_logits = rgb_logits[:, t].unsqueeze(1).expand(-1, N - t, -1)  # [B, N-t-1, C]

            # Compute negative log likelihood
            lzbaxy_nll = -torch.distributions.Normal(pred_lzbaxy_mean, pred_lzbaxy_std).log_prob(expert_lzbaxy[:,t:]).sum(-1)  # [B, N-t-1]
            rgb_nll = -torch.distributions.Categorical(logits=pred_rgb_logits).log_prob(expert_rgb_class[:,t:])  # [B, N-t-1]

            loss_matrix = lzbaxy_nll + rgb_nll  # [B, N-t-1]
            masked_loss = loss_matrix + (1 - used_mask[:,t:,0]) * 1e6  # Mask out non-existent strokes

            # Greedy match: pick minimum loss for each batch 
            softmin_weights = torch.softmax(-masked_loss, dim=1)  # [B, H_expert]
            softmin_loss = (softmin_weights * loss_matrix).sum(dim=1)  # [B]
            softmin_loss_masked = softmin_loss * mask[:,t,0]
            total_loss += softmin_loss_masked.sum()
            count += (mask[:,t,0]).sum()

            # Mark chosen strokes in the mask as invalid for future iterations
            # Clone so that loss gradients don't mix
            _, min_idx = loss_matrix.min(dim=1)
            used_mask[torch.arange(B),min_idx] = 0.0
            used_mask = used_mask.clone()
        return total_loss / count

    def set_loss(self, lzbaxy_mean, lzbaxy_std, rgb_logits, gt_strokes, mask):
        # Process expert actions into LZBAxAyXY tensor and RGB onehot
        gt_lzbaaxy, gt_rgb_onehot, _ = self.breakdown_stroke(gt_strokes)
        gt_features = torch.cat([gt_lzbaaxy, torch.zeros_like(gt_lzbaaxy), gt_rgb_onehot], dim=-1)
        gt_latent = self.stroke_encoder(gt_features*mask)

        pred_features = torch.cat([lzbaxy_mean, lzbaxy_std, rgb_logits], dim=-1)
        pred_latent = self.stroke_encoder(pred_features*mask)

        loss = 1 - F.cosine_similarity(pred_latent, gt_latent, dim=-1).mean()
        return loss
    
    def paint_example_canvas(self, canvas_name='example_canvas.png'):
        self.model.eval()
        self.model.save_hx = True
        self.model.hx = None
        state = {k:v[0,0].unsqueeze(0).clone() for k,v in self.prev_init_state.items()}
        H = int(state['remaining_strokes'][0,0].item())
        for i in range(H):
            stroke = self.model.sample(**state).unsqueeze(0)
            new_canv = self.paint_stroke(stroke, state['current_canvas'])
            new_state = {k:v.clone() for k,v in state.items()}
            new_state['current_canvas'] = new_canv
            new_state['remaining_strokes'] -= 1
            new_state['mask'] = torch.where(new_state['remaining_strokes'] <= 0, torch.zeros_like(new_state['mask']), torch.ones_like(new_state['mask'])).to(torch.float)
            state = new_state
        
        # Add target canvas to generated canvas and save
        target_canvas = state['target_img']
        output_canvas = torch.cat([target_canvas[0], torch.zeros((3,target_canvas.shape[-2],3)).to(self.device), new_canv[0]], dim=-1)
        plt.close()
        plt.imshow(output_canvas.detach().cpu().permute(1,2,0).numpy())
        plt.savefig(os.path.join(self.save_folder,canvas_name))
        self.model.save_hx = False
        self.model.hx = None
        self.model.train()
    
    def paint_stroke(self, stroke_tensors, canvas_current):
        # Model outputs [l, z, b, a, x, y, r, g, b]
        stroke_tensors = stroke_tensors.unsqueeze(-1)
        strokes = []
        for i in range(stroke_tensors.shape[1]):
            generated_stroke = BrushStrokeBatch(self.opt,
                                        stroke_length=stroke_tensors[:,i,0],
                                        stroke_z=stroke_tensors[:,i,1],
                                        stroke_bend=stroke_tensors[:,i,2],
                                        stroke_alpha=torch.zeros(1,1).to(self.device),
                                        color=stroke_tensors[:,i,6:].squeeze(-1),
                                        a=stroke_tensors[:,i,3],
                                        xt=stroke_tensors[:,i,4],
                                        yt=stroke_tensors[:,i,5],
                                        init_differentiably=True,
                                        ink=None)
            strokes.append(generated_stroke)
        generated_painting = PaintingBatch(self.opt, background_img=canvas_current).to(self.device)
        generated_canvas = generated_painting(strokes, self.h_render, self.w_render, use_alpha=False, return_alphas=False)
        return generated_canvas
import os
import clip
import torch
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm
from matplotlib import pyplot as plt

from painting import PaintingBatch
from brush_stroke import BrushStrokeBatch
from stroke_generator.IL.random_expert import RandomExpert
from stroke_generator.IL.model import StrokePredictor

class OfflineStrokeGeneratorTrainer():
    def __init__(self, opt, model: StrokePredictor, device, dataloader, save_folder='', run_name='run'):
        self.opt = opt
        self.model = model
        self.device = device
        self.dataloader = dataloader

        self.optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=100)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=10, T_mult=2)
        self.random_expert = RandomExpert(opt)

        # Canvas params
        self.n_strokes = 1 # Strokes per canvas
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

        # Save params
        self.save_folder = save_folder
        self.run_name = run_name

        # Loss params
        # Length, Z, and Bend
        l_scale = self.opt.MAX_STROKE_LENGTH - self.opt.MIN_STROKE_LENGTH
        z_scale = 0.95 - self.opt.MIN_STROKE_Z
        b_scale = self.opt.MAX_BEND
        self.stroke_scale = torch.tensor([1/l_scale, 1/z_scale, 1/b_scale, 1/(2*torch.pi), 1, 1, 1, 1, 1], device=self.device)
        self.stroke_weight = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1, 1], device=self.device)
    
    def train(self, epochs):
        epoch_losses = []
        tokenized_text = clip.tokenize(["A painting"]*self.dataloader.batch_size).detach().to(self.device)
        tokenized_text.requires_grad = False
        self.lr_scheduler.T_0 = epochs//100+1
        best_loss = 1e10

        for e in range(epochs):
            losses = []
            for i, data in tqdm(enumerate(self.dataloader), desc="Training Epoch...", unit="Batch", total=len(self.dataloader)):
                # TODO: Add multistroke
                canvas_current = data[0].to(self.device)
                canvas_target = data[1].to(self.device)
                remaining_strokes = data[2].to(self.device)
                color_palette = -1*torch.ones(canvas_current.shape[0], 12, 3).to(self.device)

                # canvas_noise = torch.randn_like(canvas_current).to(self.device)*0.01
                # canvas_current -= canvas_noise
                # canvas_target -= canvas_noise

                strokes_gt = data[3][:,0].detach().to(self.device)
                # stroke_noise = torch.randn_like(strokes_gt).to(self.device)*self.stroke_scale/100
                # strokes_gt += stroke_noise

                # Forward pass
                strokes_pred = self.model(
                    current_canvas=canvas_current,
                    target_img=canvas_target,
                    target_tokenized_txt=tokenized_text[0:canvas_current.shape[0]], # Make sure this matches in edge case of leftover batch piece
                    remaining_strokes=remaining_strokes,
                    color_palette=color_palette
                )
                # normal = torch.distributions.Normal(mu, log_std.exp())
                # strokes_pred = normal.rsample()

                # self.optim.zero_grad()
                # loss = normal.log_prob(strokes_gt).mean()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 10.0)
                # self.optim.step()

                self.optim.zero_grad()
                loss = self.loss_fn(strokes_pred, strokes_gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optim.step()

                losses.append(loss.item())
            
            if self.lr_scheduler is not None: self.lr_scheduler.step()
            epoch_losses.append(torch.tensor(losses).mean().item())
            print(f"Epoch: {e} | | Loss: {epoch_losses[-1]:0.4f}")

            # Paint ground truth brush stroke and save image after every epoch
            
            with torch.no_grad():
                # Save stroke examples
                if e%(epochs//10)==0:
                    for i in range(2):
                        self.paint_and_save(strokes_pred[i].unsqueeze(0).unsqueeze(-1), canvas_current[i].unsqueeze(0), canvas_target[i], f'_{self.run_name}_{e}_{i}')
                # Plot losses
                plt.close()
                plt.plot(epoch_losses)
                plt.savefig(os.path.join(self.save_folder,f"losses_{self.run_name}.png"))
                # Save model
                if epoch_losses[-1] < best_loss:
                    best_loss = epoch_losses[-1]
                    torch.save(self.model.state_dict(), os.path.join(self.save_folder,f"stroke_generator_state_dict_{self.run_name}.pth"))

        torch.save(torch.tensor(losses), os.path.join(self.save_folder,f"losses_{self.run_name}.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.save_folder,f"stroke_generator_state_dict_{self.run_name}.pth"))

    def loss_fn(self, y_pred, y_gt):
        # Weighted MSE that takes into account the relative scales of each parameter

        # Everything but angle (angle weight is 0)
        scaled_mse = ((y_gt - y_pred) ** 2) * self.stroke_scale * self.stroke_weight
        
        # Angle
        a_pred = y_pred[:,3]
        a_gt = y_gt[:,3]
        a_loss = ((torch.sin(a_pred) - torch.sin(a_gt)) ** 2 + (torch.cos(a_pred) - torch.cos(a_gt)) ** 2)

        total_loss = scaled_mse.mean() + a_loss.mean()
        return total_loss
    
    def paint_and_save(self, stroke, canvas_current, canvas_target, name=''):
        # Model outputs [l, z, b, a, x, y, r, g, b]
        generated_stroke = BrushStrokeBatch(self.opt,
                                    stroke_length=stroke[:,0],
                                    stroke_z=stroke[:,1],
                                    stroke_bend=stroke[:,2],
                                    stroke_alpha=torch.zeros(1,1).to(self.device),
                                    color=stroke[:,6:].squeeze(-1),
                                    a=stroke[:,3],
                                    xt=stroke[:,4],
                                    yt=stroke[:,5],
                                    init_differentiably=True,
                                    ink=None)
        generated_painting = PaintingBatch(self.opt, background_img=canvas_current).to(self.device)
        generated_canvas = generated_painting([generated_stroke], self.h_render, self.w_render, use_alpha=False, return_alphas=False)
        
        joint_canvas = torch.cat([canvas_target, torch.zeros((3,canvas_target.shape[1],3)).to(self.device), generated_canvas[0]], dim=-1)
        
        self.save_canvas_as_img(joint_canvas, path=self.save_folder, suffix=name)
    
    def save_canvas_as_img(self, canvas, path='outputs', suffix=''):
        plt.imshow(canvas.detach().cpu().permute(1,2,0).numpy())
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except:
                raise Exception(f"Path provided for saving canvas does not exist and couldn't be created.\n\t{path}")
        plt.savefig(os.path.join(path,f"canvas{suffix}.png"))



class OfflineMultiStrokeGeneratorTrainer(OfflineStrokeGeneratorTrainer):
    def __init__(self, opt, model: StrokePredictor, device, dataloader, save_folder='', run_name='run'):
        super().__init__(opt, model, device, dataloader, save_folder, run_name)
    
    def train(self, epochs):
        epoch_losses = []
        tokenized_text = clip.tokenize(["A painting"]*self.dataloader.batch_size).detach().to(self.device)
        tokenized_text.requires_grad = False
        self.lr_scheduler.T_0 = epochs//100+1
        best_loss = 1e10

        for e in range(epochs):
            losses = []
            for i, data in tqdm(enumerate(self.dataloader), desc="Training Epoch...", unit="Batch", total=len(self.dataloader)):
                # Shape batch_size, n_strokes+1, 3, h_render, w_render
                canvases = data[0].to(self.device)
                strokes_gt = data[1].detach().to(self.device)
                canvas_current = canvases[:,:-1]
                canvas_target = canvases[:,-1]
                # remaining strokes should be N,N-1...1 for size [BatchSize, N, 1]
                remaining_strokes = torch.arange(canvas_current.shape[1], 0, -1)
                remaining_strokes = remaining_strokes.unsqueeze(0).unsqueeze(-1).repeat(canvas_current.shape[0],1,1).to(self.device).to(torch.float)                 

                # Set up masks. Masked items are stored as all -1s
                invalid_slots = (canvas_current == -1).all(dim=-1).unsqueeze(-1).repeat(1,1,1,1,canvas_current.shape[-1])
                canvas_mask = torch.where(invalid_slots, torch.zeros_like(canvas_current), torch.ones_like(canvas_current))
                invalid_slots = (strokes_gt == -1).all(dim=-1).unsqueeze(-1)
                stroke_mask = torch.where(invalid_slots, torch.zeros_like(remaining_strokes), torch.ones_like(remaining_strokes))

                # repeat canvas_target for all canvas_current
                canvas_target = canvas_target.unsqueeze(1).repeat(1,canvas_current.shape[1],1,1,1)
                tok_text = tokenized_text[0:canvas_current.shape[0]].unsqueeze(1).repeat(1,canvas_current.shape[1],1)
                color_palette = -1*torch.ones(canvas_current.shape[0], canvas_current.shape[1], 12, 3).to(self.device)

                # Forward pass
                strokes_pred = self.model(
                    current_canvas=canvas_current,
                    target_img=canvas_target,
                    target_tokenized_txt=tok_text,
                    remaining_strokes=remaining_strokes,
                    color_palette=color_palette,
                    canvas_mask=canvas_mask,
                    stroke_mask=stroke_mask
                )
                # normal = torch.distributions.Normal(mu, log_std.exp())
                # strokes_pred = normal.rsample()

                # self.optim.zero_grad()
                # loss = normal.log_prob(strokes_gt).mean()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), 10.0)
                # self.optim.step()

                self.optim.zero_grad()
                loss = self.loss_fn(strokes_pred*stroke_mask, strokes_gt*stroke_mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.optim.step()

                losses.append(loss.item())
            
            if self.lr_scheduler is not None: self.lr_scheduler.step()
            epoch_losses.append(torch.tensor(losses).mean().item())
            print(f"Epoch: {e} | | Loss: {epoch_losses[-1]:0.4f}")

            # Paint ground truth brush stroke and save image after every epoch
            
            with torch.no_grad():
                # Save stroke examples
                if e%(epochs//5)==0:
                    for i in range(2):
                        n = (stroke_mask[i]==1).all(dim=-1).sum()
                        s_p = strokes_pred[i,-n:].unsqueeze(0)
                        c_c = canvas_current[i,-n:]
                        c_t = canvas_target[i,0]
                        self.paint_and_save(s_p, c_c, c_t, f'_{self.run_name}_{e}_{i}')
                # Plot losses
                plt.close()
                plt.plot(epoch_losses)
                plt.savefig(os.path.join(self.save_folder,f"losses_{self.run_name}.png"))
                # Save model
                if epoch_losses[-1] < best_loss:
                    best_loss = epoch_losses[-1]
                    torch.save(self.model.state_dict(), os.path.join(self.save_folder,f"stroke_generator_state_dict_{self.run_name}.pth"))

        torch.save(torch.tensor(losses), os.path.join(self.save_folder,f"losses_{self.run_name}.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.save_folder,f"stroke_generator_state_dict_{self.run_name}.pth"))

    def loss_fn(self, y_pred, y_gt):
        # Weighted MSE that takes into account the relative scales of each parameter

        # y_pred.shape = batch_size, 9
        # y_gt.shape = batch_size, num_strokes, 9

        # Find lowest MSE error for all gt strokes
        # Return index and loss
        scaled_mse = ((y_gt - y_pred) ** 2) * self.stroke_scale * self.stroke_weight

        # Everything but angle (angle weight is 0)
        # scaled_mse = ((y_gt - y_pred) ** 2) * self.stroke_scale * self.stroke_weight
        
        # Angle
        a_pred = y_pred[...,3]
        a_gt = y_gt[...,3]
        a_loss = ((torch.sin(a_pred) - torch.sin(a_gt)) ** 2 + (torch.cos(a_pred) - torch.cos(a_gt)) ** 2)

        total_loss = scaled_mse.mean() + a_loss.mean()
        return total_loss

    def paint_and_save(self, stroke_tensors, canvas_current, canvas_target, name=''):
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
        
        joint_canvas = torch.cat([canvas_target, torch.zeros((3,canvas_target.shape[1],3)).to(self.device), generated_canvas[0]], dim=-1)
        
        self.save_canvas_as_img(joint_canvas, path=self.save_folder, suffix=name)
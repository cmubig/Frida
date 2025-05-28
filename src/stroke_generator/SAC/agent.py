import torch
from torch import nn

from stroke_generator.SAC.actor import StrokeActor
from stroke_generator.model import StrokePredictor
from stroke_generator.SAC.critic import StrokeCritic
from stroke_generator.utils.replay_buffer import ReplayBuffer

from losses.clip_loss import CLIPConvLoss, Dict2Class
from torchvision import transforms

from brush_stroke import BrushStrokeBatch
from painting import PaintingBatch

import torch.nn.functional as F


class StrokeAgent(nn.Module):
    def __init__(self, opt, model: StrokePredictor, device, batch_size=16):
        super().__init__()

        self.opt = opt
        self.device = device

        # Canvas params
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

        self.actor = StrokeActor(opt, model, device)
        
        # Critic
        state_latent_dim = self.actor.model.state_latent_dim
        action_latent_dim = 1024
        self.critic = StrokeCritic(opt, device, state_latent_dim, action_latent_dim)
        self.target_critic = StrokeCritic(opt, device, state_latent_dim, action_latent_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optims
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)
        self.critic_optim = torch.optim.AdamW(list(self.critic.parameters()) + 
                                              list(self.actor.model.state_encoder.parameters()), 
                                              lr=1e-3)

        # Loss funcs
        clip_conv_layer_weights = [0, 0, 0, 0, 1.0]
        a = {'clip_model_name':'ViT-B/32','clip_conv_loss_type':'Cos','device':device,
            'num_aug_clip':10,'augemntations':['affine'],
            'clip_fc_loss_weight':0.0,'clip_conv_layer_weights':clip_conv_layer_weights}
        self.clip_conv_loss_model = CLIPConvLoss(Dict2Class(a))
        self.blur = transforms.GaussianBlur(7,sigma=1.0)

        l_min, l_max = self.opt.MIN_STROKE_LENGTH, self.opt.MAX_STROKE_LENGTH
        z_min, z_max = self.opt.MIN_STROKE_Z, 0.95
        b_scale = self.opt.MAX_BEND
        self.action_min = torch.tensor([l_min, z_min, -b_scale, -3.14, -1.0, -1.0, 0.0, 0.0, 0.0]).to(self.device)
        self.action_max = torch.tensor([l_max, z_max, b_scale, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)

        self.rollouts = 3
        self.horizon = 15
        self.batch_size = batch_size
        self.actor_frozen = False

        # Reward weights
        self.mse_improvement_w = 100.0
        self.mse_w = 10.0
        self.action_diff_w = 1.0
        self.action_in_bounds_w = 3.0
        self.paint_vol_w = 100.0
        self.paint_overlap_w = 100.0
        self.clip_w = 0.1

        self.gamma = 0.9 # discount factor
        self.tau = 5e-3  # soft target update parameter
        self.eps = 0.1 # exploration parameter

        # Learnable alpha
        self.target_entropy = -torch.tensor(7.0).to(device)  # 7 continuous, 1 categorical (RGB)
        self.log_alpha = torch.tensor(-20.0, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=5e-4)
    
    # state = dict{current_canvas, target_img, target_tokenized_txt, remaining_strokes, color_palette, mask}
    def plan_action(self, state, hidden_cell_actor = None, hidden_cell_critic = (None, None)):
        _z0 = self.actor.model.state_encoder(**state)
        
        actions = torch.zeros(self.rollouts, _z0.shape[0], 9).to(self.device)
        hiddens_cells_actor = torch.zeros(self.rollouts, 2, *hidden_cell_actor[0].shape).to(self.device)
        hiddens_cells_critic = torch.zeros(self.rollouts, 4, *hidden_cell_critic[0][0].shape).to(self.device)
        returns = torch.zeros(self.rollouts, _z0.shape[0]).to(self.device)
        
        for i in range(self.rollouts):
            hc_a, hc_c = hidden_cell_actor, hidden_cell_critic
            _z = _z0
            for j in range(self.horizon):
                # Sample action
                _a_enc, hc_a = self.actor(_z, hc_a)
                _a_enc = _a_enc.squeeze(1)
                
                # Apply action
                action = self.actor.decode_and_sample_action(_a_enc, state['color_palette'])
                if torch.rand(1) < self.eps: # Eps greedy random action
                    scale = torch.cat([self.actor.model.scale[:3],torch.ones(1).to(self.device),self.actor.model.scale[-2:]])
                    bias = torch.cat([self.actor.model.bias[:3],torch.zeros(1).to(self.device),self.actor.model.bias[-2:]])
                    lzbaxy = torch.rand_like(action[:,:-3]) * scale + bias
                    rgb = state['color_palette'][torch.arange(lzbaxy.shape[0]),torch.randint(12,(lzbaxy.shape[0],))]
                    action = torch.cat([lzbaxy, rgb], dim=-1)
                new_state = self.apply_action(state, action)

                # Encode action and new state
                _a_enc = self.critic.stroke_encoder(action)
                _new_z = self.actor.model.state_encoder(**new_state)

                # Use Q value estimate
                q1, q2, hc_c = self.critic(_new_z, _a_enc, hc_c)
                q_val = torch.min(q1, q2)
                returns[i] += (self.gamma**j * q_val * state['mask']).squeeze(-1)

                # Update state
                if j == 0:
                    actions[i] = action
                    hiddens_cells_actor[i,0] = hc_a[0]
                    hiddens_cells_actor[i,1] = hc_a[1]
                    hiddens_cells_critic[i,0] = hc_c[0][0]
                    hiddens_cells_critic[i,1] = hc_c[0][1]
                    hiddens_cells_critic[i,2] = hc_c[1][0]
                    hiddens_cells_critic[i,3] = hc_c[1][1]
                _z = _new_z
                
                # Break if all done
                if new_state['mask'].sum() == 0:
                    break
        
        # Select action with highest estimated return
        best_action_idx = torch.argmax(returns, dim=0)
        batch_indices = torch.arange(actions.shape[1], device=actions.device)
        action = actions[best_action_idx, batch_indices]
        hc_a = (
            hiddens_cells_actor[best_action_idx, 0, :, batch_indices].permute(1,0,2).contiguous(),
            hiddens_cells_actor[best_action_idx, 1, :, batch_indices].permute(1,0,2).contiguous(),
        )
        hc_c = (
            (
                hiddens_cells_critic[best_action_idx, 0, :, batch_indices].permute(1,0,2).contiguous(),
                hiddens_cells_critic[best_action_idx, 1, :, batch_indices].permute(1,0,2).contiguous()
            ),
            (
                hiddens_cells_critic[best_action_idx, 2, :, batch_indices].permute(1,0,2).contiguous(),
                hiddens_cells_critic[best_action_idx, 3, :, batch_indices].permute(1,0,2).contiguous()
            )
        )
        return action, hc_a, hc_c

    def apply_action(self, state, action):
        action = torch.clamp(action, self.action_min, self.action_max).unsqueeze(-1)
        stroke = BrushStrokeBatch(
            self.opt,
            stroke_length=action[:,0],
            stroke_z=action[:,1],
            stroke_bend=action[:,2],
            stroke_alpha=torch.zeros(action.shape[0],1).to(self.device),
            color=action[:,6:].squeeze(-1),
            a=action[:,3],
            xt=action[:,4],
            yt=action[:,5],
            init_differentiably=True,
            ink=None
        )

        painting = PaintingBatch(self.opt, background_img=state['current_canvas']).to(self.device)
        updated_canvas = painting([stroke], self.h_render, self.w_render, use_alpha=False, return_alphas=False)

        new_state = {k: v.clone() for k, v in state.items()}
        m = state['mask'].squeeze(-1).to(torch.bool)
        new_state['current_canvas'][m] = updated_canvas[m]
        new_state['remaining_strokes'][m] -= 1
        new_state['mask'] = torch.where(new_state['remaining_strokes'] > 0, torch.ones_like(state['mask']), torch.zeros_like(state['mask']))
        return new_state

    def evaluate_state(self, state, prev_state, action, prev_action):

        # % MSE improvement
        # Reward is based on how much the MSE decreased
        mse_loss = F.mse_loss(state['current_canvas'], state['target_img'], reduction='none').mean(dim=(-1,-2,-3))
        prev_mse_loss = F.mse_loss(prev_state['current_canvas'], prev_state['target_img'], reduction='none').mean(dim=(-1,-2,-3))
        mse_improvement_reward = ((prev_mse_loss - mse_loss)/(prev_mse_loss+1e-6)).clamp(min=0)
        
        # MSE
        # Gaussian blur to help with proximity
        # Masking to only do MSE where there's a blurry target stroke
        blurred_current_canvas = self.blur(state['current_canvas'].clone())
        blurred_target_img = self.blur(state['target_img'].clone())
        blurry_mse_loss = F.mse_loss(blurred_current_canvas, blurred_target_img, reduction='none')
        blurry_masked_mse_loss = torch.where(blurred_target_img<0.99, blurry_mse_loss, torch.zeros_like(blurry_mse_loss))
        mse_reward = -blurry_masked_mse_loss.mean(dim=(-1,-2,-3))

        # Paint Volume Matching
        # Reward is based on how similar the volume of canvas with paint on it is
        target_has_paint = (state['target_img'] < 0.99).any(dim=-3)
        current_has_paint = (state['current_canvas'] < 0.99).any(dim=-3)
        painted_vol_ratio_target = target_has_paint.sum(dim=(-1,-2)) / (self.h_render * self.w_render * 3)
        painted_vol_ratio_current = current_has_paint.sum(dim=(-1,-2)) / (self.h_render * self.w_render * 3)
        painted_vol_reward = -(painted_vol_ratio_target - painted_vol_ratio_current)**2

        # Paint Overlap Matching
        blurry_target_has_paint = (blurred_target_img < 0.99).any(dim=-3)
        blurry_current_has_paint = (blurred_current_canvas < 0.99).any(dim=-3)
        clear_paint_overlap_reward = torch.logical_and(target_has_paint, current_has_paint).to(torch.float).sum(dim=(-1,-2))/target_has_paint.sum(dim=(-1,-2))
        blurry_paint_overlap_reward = torch.logical_and(blurry_target_has_paint, blurry_current_has_paint).to(torch.float).sum(dim=(-1,-2))/blurry_target_has_paint.sum(dim=(-1,-2))
        paint_overlap_reward = clear_paint_overlap_reward + 0.2*blurry_paint_overlap_reward
        paint_overlap_reward[paint_overlap_reward.isnan()] = 0.0

        # CLIP reward
        # Reward is based on how similar the CLIP features of the current canvas and target image are
        clip_reward = torch.zeros_like(mse_reward)
        for i in range(mse_reward.shape[0]):
            current = state['current_canvas'][i].unsqueeze(0)
            target = state['target_img'][i].unsqueeze(0)
            clip_losses = self.clip_conv_loss_model(current, target)
            clip_loss = 0
            for key in clip_losses.keys():
                clip_loss += clip_losses[key]
            clip_reward[i] = -1 * clip_loss
        
        state_reward = self.mse_improvement_w * mse_improvement_reward + \
                        self.mse_w * mse_reward + \
                        self.paint_vol_w * painted_vol_reward + \
                        self.paint_overlap_w * paint_overlap_reward + \
                        self.clip_w * clip_reward
        

        # Action diversity reward
        # Reward is based on how different the action is from the previous action
        # Clamped to avoid forcing extreme action diversity
        action_weights = torch.tensor([1,1,1,1,10,10,0,0,0]).to(self.device)
        action_diff_reward = F.mse_loss(action*action_weights, prev_action*action_weights, reduction='none').mean(dim=-1).clamp(max=0.2)

        # Action in bounds reward
        action_weights = torch.tensor([1,1,1,1,1,1,0,0,0]).to(self.device)
        is_valid = torch.logical_and(action > self.action_min, action < self.action_max)
        action_in_bounds_reward = (is_valid*action_weights).sum(dim=-1).to(torch.float)

        action_reward = self.action_diff_w * action_diff_reward + \
                        self.action_in_bounds_w * action_in_bounds_reward
        
        total_reward = state_reward + action_reward
        return total_reward



    def update(self, replay_buffer: ReplayBuffer):
        state, hidden_cell_actor, hidden_cell_critic, action, reward, next_state, done = replay_buffer.sample(self.batch_size, self.device)
        # state = dict{current_canvas, target_img, target_tokenized_txt, remaining_strokes, color_palette, mask}

        mask = state['mask'].squeeze(-1).to(torch.bool)

        # Encode state, action, and next states and apply masks to everything
        _s = self.actor.model.state_encoder(**state)[mask]
        _a = self.critic.stroke_encoder(action)[mask]
        _ns = self.actor.model.state_encoder(**next_state)[mask]
        hc_a = (hidden_cell_actor[0][:,mask], hidden_cell_actor[1][:,mask])
        hc_c = (
            (hidden_cell_critic[0][0][:,mask], hidden_cell_critic[0][1][:,mask]),
            (hidden_cell_critic[1][0][:,mask], hidden_cell_critic[1][1][:,mask])
        )
        color_palette = state['color_palette'][mask]
        r = reward[mask]
        d = done[mask]

        ### Update Critic ###
        with torch.no_grad():
            _na_enc, _ = self.actor(_s, hc_a)
            lzbaxy_mu, lzbaxy_log_std, rgb_logits = self.actor.decode_action(_na_enc)

            lzbaxy_std = lzbaxy_log_std.exp()
            next_lzbaxy_dist = torch.distributions.Normal(lzbaxy_mu, lzbaxy_std)
            next_lzbaxy = next_lzbaxy_dist.rsample()
            rgb_dist = torch.distributions.Categorical(logits=rgb_logits)
            rgb_idx = rgb_dist.sample()
            next_rgb = color_palette[torch.arange(rgb_idx.shape[0]), rgb_idx]
            next_action = torch.cat(
                [next_lzbaxy[:,0:3],
                 torch.atan2(next_lzbaxy[:,3], next_lzbaxy[:,4]).unsqueeze(-1),
                 next_lzbaxy[:,-2:],
                  next_rgb], dim=-1)
            _na = self.critic.stroke_encoder(next_action)

            sample_log_prob = (next_lzbaxy_dist.log_prob(next_lzbaxy).sum(dim=-1, keepdim=True) + rgb_dist.log_prob(rgb_idx).unsqueeze(-1))

            target_q1, target_q2, _ = self.target_critic(_ns, _na, hc_c)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * sample_log_prob
            target_value = r + self.gamma * (1 - d.to(torch.float)) * target_q
        
        q1, q2, _ = self.critic(_s, _a, hc_c)
        q1_reg, q2_reg, target_reg = q1/1.0, q2/1.0, target_value/1.0
        critic_loss = nn.functional.mse_loss(q1_reg, target_reg) + nn.functional.mse_loss(q2_reg, target_reg)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        ### Update Actor ###
        if not self.actor_frozen:
            # Sample action
            _s = self.actor.model.state_encoder(**state) # Recalculate for graph purposes
            _a, _ = self.actor(_s, hc_a)
            action_sample, log_prob = self.sample_and_log_prob(_s, state['color_palette'])
            _a_enc = self.critic.stroke_encoder(action_sample)

            # Update alpha (from SAC v2)
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            # calculate Q value for the sampled action
            q1, q2, _ = self.critic(_s, _a_enc, hc_c)

            # Calculate grad J
            # log_prob = self.actor_log_prob(state, _s, action_sample)[mask]
            Q = torch.min(q1, q2).detach()
            alpha = self.log_alpha.exp()
            log_prob, Q = log_prob[mask], Q[mask] # Apply masks
            actor_loss = (alpha * log_prob - Q).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        ### Update Target Critic ###
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def sample_and_log_prob(self, encoded_state, color_palette):
         # Get latent action distribution params
        latent_action, _ = self.actor(encoded_state)
        lzbaxy_mu, lzbaxy_log_std, rgb_logits = self.actor.decode_action(latent_action)
        lzbaxy_std = lzbaxy_log_std.exp()

        # Sample from Gaussian
        normal = torch.distributions.Normal(lzbaxy_mu, lzbaxy_std)
        z = normal.rsample()  
        lzbaxy_sample = torch.tanh(z) # Reparameterized sample
        
        # # Split up the lzbaxy_mu into individual components
        # lzb_mu, lzb_std = lzbaxy_mu[:,0:3], lzbaxy_std[:,0:3]
        # lzb_sample = lzbaxy_sample[:,0:3]
        # a_mu, a_std = lzbaxy_mu[:,3:5], lzbaxy_std[:,3:5]
        # a_sample = lzbaxy_sample[:,3]
        # xy_mu, xy_std = lzbaxy_mu[:,5:], lzbaxy_std[:,5:]
        # xy_sample = lzbaxy_sample[:,4:6]

        # Log prob with tanh correction
        log_prob = normal.log_prob(z).sum(dim=-1)
        log_prob -= torch.log(torch.clamp(1 - lzbaxy_sample.pow(2), min=1e-6)).sum(dim=-1)

        # Sample RGB from categorical
        rgb_dist = torch.distributions.Categorical(logits=rgb_logits)
        rgb_idx = rgb_dist.sample()
        rgb_log_prob = rgb_dist.log_prob(rgb_idx)

        total_log_prob = log_prob + rgb_log_prob

        # Get action
        rgb = color_palette[torch.arange(rgb_idx.shape[0]), rgb_idx]
        action = torch.cat([lzbaxy_sample[:,0:3],
                            torch.atan2(lzbaxy_sample[:,3], lzbaxy_sample[:,4]).unsqueeze(-1),
                            lzbaxy_sample[:,5:],
                            rgb], dim=-1)
        
        return action, total_log_prob

        # # Length, Z, Bend loss
        # lzb_dist = torch.distributions.Normal(lzb_mu, lzb_std)
        # lzb_log_prob = lzb_dist.log_prob(lzb_sample).mean(dim=-1)

        # # Breaks down angle into X and Y components to avoid discontinuities
        # a_xy_gt = torch.cat([torch.sin(a_gt).unsqueeze(-1), torch.cos(a_gt).unsqueeze(-1)],dim=-1)
        # a_dist = torch.distributions.Normal(a_mu, a_std)
        # a_log_prob = a_dist.log_prob(a_xy_gt).mean(dim=-1)

        # # X and Y loss
        # xy_dist = torch.distributions.Normal(xy_mu, xy_std)
        # xy_log_prob = xy_dist.log_prob(xy_gt).mean(dim=-1)

        # # RGB loss (Categorical b/c of palette)
        # rgb_dist = torch.distributions.Categorical(logits=rgb_logits)  # Categorical distribution
        # rgb_onehot = (action[:,-3:].unsqueeze(-2).repeat(1,12,1)==state['color_palette']).all(dim=-1).to(torch.float)
        # rgb_log_prob = rgb_dist.log_prob(rgb_onehot.argmax(dim=-1))

        # return lzb_log_prob + a_log_prob + xy_log_prob + rgb_log_prob
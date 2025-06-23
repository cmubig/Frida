import clip
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from losses.clip_loss import CLIPConvLoss, Dict2Class
from brush_stroke import BrushStrokeBatch
from painting import PaintingBatch

from stroke_generator.utils.model_utils import print_memory_update

class StrokeEnv:
    def __init__(self, opt, device, num_envs,
                 target_expert, max_strokes=10):
        self.opt = opt
        self.device = device
        self.num_envs = num_envs
        self.target_expert = target_expert
        self.max_strokes = max_strokes

        l_min, l_max = self.opt.MIN_STROKE_LENGTH, self.opt.MAX_STROKE_LENGTH
        z_min, z_max = self.opt.MIN_STROKE_Z, 0.95
        b_scale = self.opt.MAX_BEND
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
        self.action_min = torch.tensor([l_min, z_min, -b_scale, -3.14, -1.0, -1.0, 0.0, 0.0, 0.0]).to(self.device)
        self.action_max = torch.tensor([l_max, z_max, b_scale, 3.14, 1.0, 1.0, 1.0, 1.0, 1.0]).to(self.device)

        
        # Initial states
        self.starting_canvas = torch.ones(self.num_envs, 3, self.h_render, self.w_render).to(self.device)
        self.tokenized_text = clip.tokenize(["A splash of colors on a white background"]*self.num_envs).detach().to(self.device)
        self.state = {
            "current_canvas": torch.zeros(self.num_envs, 3, self.h_render, self.w_render).to(self.device),
            "target_img": torch.ones(self.num_envs, 3, self.h_render, self.w_render).to(self.device),
            "target_tokenized_txt": clip.tokenize(["Sample text"]*num_envs).detach().to(self.device),
            "remaining_strokes": torch.ones(self.num_envs, 1).to(self.device),
            "color_palette": torch.zeros(self.num_envs, 12, 3).to(self.device),
            "mask": torch.ones(self.num_envs,1).to(self.device),
        }
        self.prev_initial_state = {k:v.clone() for k,v in self.state.items()}
        self.prev_action = torch.zeros(self.num_envs, 9).to(self.device)
        self.generate_new_target = True
        
        # Reward weights
        self.mse_improvement_w = 10000.0
        self.mse_w = 1.0
        self.action_diff_w = 1.0
        self.action_in_bounds_w = 10.0
        self.paint_vol_w = 10.0
        self.paint_overlap_w = 10.0
        self.paint_com_w = 0.0001
        self.clip_w = 1.0

        # Reward functions
        # Loss funcs
        clip_conv_layer_weights = [0, 0, 0, 0, 1.0]
        a = {'clip_model_name':'ViT-B/32','clip_conv_loss_type':'Cos','device':device,
            'num_aug_clip':10,'augemntations':['affine'],
            'clip_fc_loss_weight':0.0,'clip_conv_layer_weights':clip_conv_layer_weights}
        self.clip_conv_loss_model = CLIPConvLoss(Dict2Class(a))
        self.blur = transforms.GaussianBlur(7,sigma=1.0)

    def reset(self, envs=None, strokes_remaining=None):

        if envs is None:
            envs = torch.arange(self.num_envs).to(self.device)
        
        if self.generate_new_target or len(envs) < self.num_envs:
            # Reset environment(s) to blank canvas and random color palette
            with torch.no_grad():
                self.generate_target(envs, strokes_remaining=strokes_remaining)
            
        else:
            # Reuse the previous target
            self.state = {k:v.to(self.device).clone() for k,v in self.prev_initial_state.items()}

        # Previous state and action
        for k in self.state.keys():
            temp = self.prev_initial_state[k].clone()
            temp[envs] = self.state[k][envs].detach()
            self.prev_initial_state[k] = temp
        self.prev_action[envs] = torch.zeros(len(envs), 9).to(self.device)

        return self.state
    
    def generate_target(self, envs, strokes_remaining=None):
        starting_canvas = self.starting_canvas[envs].clone()
        N = len(envs)
        color_palette = torch.rand(N, 12, 3).to(self.device)
        if strokes_remaining is None:
            remaining_strokes = torch.randint(1, self.max_strokes+1, (N,1)).to(self.device).to(torch.float)
        else:
            remaining_strokes = strokes_remaining * torch.ones((N,1)).to(self.device).to(torch.float)
        with torch.no_grad():
            _, gt_canvases = self.target_expert.rollout_trajectory(remaining_strokes.clone(), starting_canvas.clone(), color_palette.clone())
        target_img = gt_canvases[torch.arange(N), remaining_strokes[:,0].to(torch.long)].detach().clone()
        tokenized_text = self.tokenized_text[envs].detach().clone()
        mask = torch.ones(N, 1).to(self.device)

        generated_state = {
            "current_canvas": starting_canvas,
            "target_img": target_img,
            "target_tokenized_txt": tokenized_text,
            "remaining_strokes": remaining_strokes,
            "color_palette": color_palette,
            "mask": mask,
        }

        # Update envs
        if self.state is None or N == self.num_envs:
            self.state = {k:v.detach().clone() for k,v in generated_state.items()}
        else:
            for k,v in generated_state.items():
                temp = self.state[k].clone()
                temp[envs] = v.detach()
                self.state[k] = temp
        
        # Clear expert memory
        self.target_expert.clear_memory()

    def step(self, action, store_grads=False):
        new_state = self.apply_action(self.state, action)
        done = new_state['remaining_strokes'] <= 0
        mask = self.state['mask'].to(torch.bool)

        # Compute reward with graph through new_state and detached prev_state
        state_reward = self.evaluate_state(new_state, self.state).unsqueeze(-1)
        action_reward = self.evaluate_action(action, self.prev_action).unsqueeze(-1)
        reward = state_reward + action_reward

        if not store_grads:
            self.clear_grads_and_finish_step(action, new_state)

        return new_state, reward, done, mask

    # Call this independantly you called step with store_grads=True
    # Used for first order sim losses
    def clear_grads_and_finish_step(self, action, next_state):
        self.prev_action = action.detach().clone()
        self.state.clear()
        self.state = {k: v.detach().clone() for k, v in next_state.items()}


    
    def apply_action(self, state, action):
        action = torch.clamp(action, self.action_min, self.action_max).unsqueeze(-1)

        stroke = BrushStrokeBatch(
            self.opt,
            stroke_length=action[:, 0],
            stroke_z=action[:, 1],
            stroke_bend=action[:, 2],
            stroke_alpha=torch.zeros(action.shape[0], 1, device=self.device),
            color=action[:, 6:].squeeze(-1),
            a=action[:, 3],
            xt=action[:, 4],
            yt=action[:, 5],
            init_differentiably=True,
            ink=None
        )

        painting = PaintingBatch(self.opt, background_img=state['current_canvas']).to(self.device)
        updated_canvas = painting([stroke], self.h_render, self.w_render, use_alpha=False, return_alphas=False)

        # Create a clean copy of state without shared storage
        new_state = {k: v.clone() for k, v in state.items()}

        # Build safe mask
        m = new_state['mask'].squeeze(-1).to(torch.bool)         # [B]
        mask3d = m[:, None, None, None]                          # [B, 1, 1, 1]

        # Update canvas without in-place op
        new_state['current_canvas'] = torch.where(mask3d, updated_canvas, new_state['current_canvas']).contiguous()

        # Update remaining_strokes without in-place indexing
        remaining_strokes = new_state['remaining_strokes']
        delta = torch.zeros_like(remaining_strokes)
        delta[m] = 1.0
        new_state['remaining_strokes'] = (remaining_strokes - delta).contiguous()

        # Update mask
        new_state['mask'] = (new_state['remaining_strokes'] > 0).float()

        return new_state

    def evaluate_state(self, state, prev_state):

        # MSE improvement
        # Reward is based on how much the MSE decreased
        mse_loss = F.mse_loss(state['current_canvas'], 
                              state['target_img'], 
                              reduction='none').mean(dim=(-1,-2,-3))
        prev_mse_loss = F.mse_loss(prev_state['current_canvas'], 
                                   prev_state['target_img'], 
                                   reduction='none').mean(dim=(-1,-2,-3))
        mse_improvement_reward = ((prev_mse_loss-mse_loss)/prev_mse_loss).clamp(min=0.0)
        
        # MSE
        # Gaussian blur to help with proximity
        blurred_current_canvas = self.blur(state['current_canvas'].clone())
        blurred_target_img = self.blur(state['target_img'].clone())
        blurry_mse_loss = F.mse_loss(blurred_current_canvas, 
                                     blurred_target_img, 
                                     reduction='none').mean(dim=(-1,-2,-3))
        mse_reward = -blurry_mse_loss

        # Find where paintings have paint
        target_has_paint = torch.sigmoid(200 * (0.99 - state['target_img'])).mean(dim=1)
        current_has_paint = torch.sigmoid(200 * (0.99 - state['current_canvas'])).mean(dim=1)
        
        # Paint Volume Matching
        # Reward is based on how similar the volume of canvas with paint on it is
        painted_vol_ratio_target = target_has_paint.mean(dim=(-1,-2))
        painted_vol_ratio_current = current_has_paint.mean(dim=(-1,-2))
        painted_vol_reward = -((painted_vol_ratio_target - painted_vol_ratio_current)**2)/(painted_vol_ratio_target**2)

        # Paint Overlap Matching
        paint_overlap_reward = (target_has_paint*current_has_paint).sum(dim=(-1,-2)) / (target_has_paint.sum(dim=(1,2))+1e-3)

        # Paint COM
        # Compares center of masses of painted areas to encourage painting in the right area (not off canvas)
        B, H, W = target_has_paint.shape
        y_coords = torch.arange(H, device=self.device).float().view(1, H, 1)
        x_coords = torch.arange(W, device=self.device).float().view(1, 1, W)
        target_mass = target_has_paint.sum(dim=(1,2)) + 1e-6
        target_com_y = (target_has_paint * y_coords).sum(dim=(1,2)) / target_mass
        target_com_x = (target_has_paint * x_coords).sum(dim=(1,2)) / target_mass
        current_mass = current_has_paint.sum(dim=(1,2)) + 1e-6
        current_com_y = (current_has_paint * y_coords).sum(dim=(1,2)) / current_mass
        current_com_x = (current_has_paint * x_coords).sum(dim=(1,2)) / current_mass
        sq_distance = (target_com_y - current_com_y)**2 + (target_com_x - current_com_x)**2
        paint_com_reward = -sq_distance

        # CLIP reward
        # Reward is based on similarity of CLIP features
        clip_reward = torch.zeros_like(mse_reward)
        # for i in range(mse_reward.shape[0]):
        #     current = state['current_canvas'][i].unsqueeze(0)
        #     target = state['target_img'][i].unsqueeze(0)
        #     clip_losses = self.clip_conv_loss_model(current, target)
        #     clip_loss = 0
        #     for key in clip_losses.keys():
        #         clip_loss += clip_losses[key]
        #     clip_reward[i] = -1 * clip_loss
        
        state_reward = self.mse_improvement_w * mse_improvement_reward + \
                        self.mse_w * mse_reward + \
                        self.paint_vol_w * painted_vol_reward + \
                        self.paint_overlap_w * paint_overlap_reward + \
                        self.paint_com_w * paint_com_reward + \
                        self.clip_w * clip_reward

        return state_reward
        

    def evaluate_action(self, action, prev_action):
        #                             [l,z,b,a,x, y, r,g,b]
        action_weights = torch.tensor([1,1,1,1,10,10,0,0,0]).to(self.device)
        
        # Action diversity reward
        # Reward is based on how different the action is from the previous action
        if prev_action.sum() == 0:
            action_diff_reward = torch.zeros(action.shape[0], device=self.device)
        else:
            action_diff_reward = F.mse_loss(action*action_weights, 
                                            prev_action*action_weights, 
                                reduction='none').sum(dim=-1).clamp(max=10.0)
            action_diff_reward = action_diff_reward / action_weights.sum()

        # Action in bounds reward
        # Reward is negative penalty for actions that are out of bounds
        is_invalid_min = torch.sigmoid(100 * (self.action_min-action))
        is_invalid_max = torch.sigmoid(100 * (action-self.action_max))
        is_invalid = is_invalid_min+is_invalid_max
        action_in_bounds_reward = -((is_invalid*action_weights).sum(dim=-1)/action_weights.sum())

        action_reward = self.action_diff_w * action_diff_reward + \
                        self.action_in_bounds_w * action_in_bounds_reward
        
        return action_reward
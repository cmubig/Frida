import torch

from stroke_generator.SAC.agent import StrokeAgent
from stroke_generator.utils.replay_buffer import ReplayBuffer
from stroke_generator.IL.random_expert import RandomExpert
from brush_stroke import BrushStrokeBatch
from painting import PaintingBatch

class SACTrainer():
    def __init__(self, opt, device='cpu', num_envs=4, buffer_size=10000):
        self.opt = opt
        self.device = device
        self.num_envs = num_envs

        self.canv_h = int(opt.render_height)
        self.canv_w = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
        
        self.agent = StrokeAgent(opt, device)
        self.random_expert = RandomExpert(opt)
        self.buffer = ReplayBuffer(buffer_size)

        self.state = {
            "current_canvas": torch.zeros(self.num_envs, 3, self.canv_h, self.canv_w).to(self.device),
            "target_img": torch.ones(self.num_envs, 3, self.canv_h, self.canv_w).to(self.device),
            "target_tokenized_txt": None,
            "remaining_strokes": torch.ones(self.num_envs, 1).to(self.device),
            "color_palette": None,
        }

        self.gt_strokes = torch.zeros(self.num_envs, self.n_strokes, 9).to(self.device)
        self.gt_canvases = torch.zeros(self.num_envs, self.n_strokes+1, 3, self.canv_h, self.canv_w).to(self.device)
    
    def env_reset(self):
        current_canvas = torch.zeros(self.num_envs, 3, self.canv_h, self.canv_w).to(self.device)
        self.gt_strokes, self.gt_canvases = self.random_expert.rollout_trajectory(self.num_envs, self.current_canvas)
        target_img = self.gt_canvases[:, -1]
        remaining_strokes = self.gt_strokes.shape[1]
        color_palette = None

        self.state = {
            "current_canvas": current_canvas,
            "target_img": target_img,
            "target_tokenized_txt": None,
            "remaining_strokes": remaining_strokes,
            "color_palette": color_palette
        }
    
    def env_step(self):
        action = self.agent.sample_action(self.state)

        stroke = BrushStrokeBatch(
            self.opt,
            stroke_length=action[:,0],
            stroke_z=action[:,1],
            stroke_bend=action[:,2],
            stroke_alpha=torch.zeros(1,1).to(self.device),
            color=action[:,6:].squeeze(-1),
            a=action[:,3],
            xt=action[:,4],
            yt=action[:,5],
            init_differentiably=True,
            ink=None
        )

        painting = PaintingBatch(self.opt, background_img=self.current_canvas).to(self.device)
        updated_canvas = painting([stroke], self.h_render, self.w_render, use_alpha=False, return_alphas=False)

        reward = self.compute_reward(updated_canvas, self.target_img)
        done = self.remaining_strokes == 0

        next_state = {
            'current_canvas': updated_canvas,
            'target_img': self.target_img,
            'target_tokenized_txt': self.target_tokenized_txt,
            'remaining_strokes': self.remaining_strokes-1,
            'color_palette': self.color_palette
        }

        self.buffer.push(self.state, action, reward, next_state, done)
        self.state = next_state


    def train(self, rollouts=1000):
        for i in range(rollouts):
            self.env_reset()
            while not self.state['remaining_strokes'] == 0:
                self.env_step()
            if i > self.seed_rollouts:
                self.agent.update(self.buffer)
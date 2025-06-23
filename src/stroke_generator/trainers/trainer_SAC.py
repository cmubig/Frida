import os
import clip
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from stroke_generator.SAC.agent import StrokeAgent
from stroke_generator.model import StrokePredictor
from stroke_generator.SAC.random_expert import RandomExpert
from stroke_generator.utils.replay_buffer import ReplayBuffer

class SACTrainer():
    def __init__(self, opt, model: StrokePredictor, device, num_envs=4, buffer_size=10000, save_folder=''):
        self.opt = opt
        self.device = device
        self.num_envs = num_envs

        self.canv_h = int(opt.render_height)
        self.canv_w = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
        
        self.agent = StrokeAgent(opt, model, device, batch_size=16)
        self.random_expert = RandomExpert(opt, device)
        self.buffer = ReplayBuffer(buffer_size)

        self.seed_rollouts = 10
        self.update_freq = 10
        self.generate_new_target_freq = 5

        self.save_folder = save_folder

        self.starting_canvas = torch.ones(self.num_envs, 3, self.canv_h, self.canv_w).to(self.device)
        self.tokenized_text = clip.tokenize(["A splash of colors on a white background"]*self.num_envs).detach().to(self.device)
        self.state = {
            "current_canvas": torch.zeros(self.num_envs, 3, self.canv_h, self.canv_w).to(self.device),
            "target_img": torch.ones(self.num_envs, 3, self.canv_h, self.canv_w).to(self.device),
            "target_tokenized_txt": clip.tokenize(["Sample text"]*num_envs).detach().to(self.device),
            "remaining_strokes": torch.ones(self.num_envs, 1).to(self.device),
            "color_palette": torch.zeros(self.num_envs, self.agent.actor.model.palette_size, 3).to(self.device),
            "mask": torch.ones(self.num_envs,).to(self.device),
        }
        self.prev_initial_state = {k:v.clone() for k,v in self.state.items()}
        self.hidden_cell_actor = (
            torch.zeros(self.agent.actor.model.main.num_layers, self.num_envs, self.agent.actor.model.main.hidden_size).to(self.device),
            torch.zeros(self.agent.actor.model.main.num_layers, self.num_envs, self.agent.actor.model.main.hidden_size).to(self.device)
        )
        self.hidden_cell_critic = (
            (torch.zeros(self.agent.critic.q1.LSTM.num_layers, self.num_envs, self.agent.critic.hidden_size).to(self.device),
             torch.zeros(self.agent.critic.q1.LSTM.num_layers, self.num_envs, self.agent.critic.hidden_size).to(self.device)),
            (torch.zeros(self.agent.critic.q2.LSTM.num_layers, self.num_envs, self.agent.critic.hidden_size).to(self.device),
            torch.zeros(self.agent.critic.q2.LSTM.num_layers, self.num_envs, self.agent.critic.hidden_size).to(self.device))
        )
        self.generate_new_target = True
        self.prev_action = torch.zeros(self.num_envs, 9).to(self.device)

        self.max_strokes = 100
        self.update_returns = []
        self.return_history = []
    
    def env_reset(self):
        # Reset environment to blank canvas and random color palette
        if self.generate_new_target:
            starting_canvas = self.starting_canvas.clone()
            color_palette = torch.rand(self.num_envs, 12, 3).to(self.device)
            remaining_strokes = torch.randint(1, self.max_strokes+1, (self.num_envs,)).unsqueeze(-1).to(self.device).to(torch.float)
            _, gt_canvases = self.random_expert.rollout_trajectory(remaining_strokes.clone(), starting_canvas.clone(), color_palette.clone())
            target_img = gt_canvases[torch.arange(self.num_envs), remaining_strokes[:,0].to(torch.long)].clone()
            tokenized_text = self.tokenized_text.clone()
            mask = torch.ones(self.num_envs, 1).to(self.device)

            self.state = {
                "current_canvas": starting_canvas,
                "target_img": target_img,
                "target_tokenized_txt": tokenized_text,
                "remaining_strokes": remaining_strokes,
                "color_palette": color_palette,
                "mask": mask,
            }
        else:
            self.state = {k:v.clone() for k,v in self.prev_initial_state.items()}

        self.prev_initial_state = {k:v.clone() for k,v in self.state.items()}
        self.hidden_cell_actor = (
            torch.zeros(self.agent.actor.model.main.num_layers, self.num_envs, self.agent.actor.model.main.hidden_size).to(self.device),
            torch.zeros(self.agent.actor.model.main.num_layers, self.num_envs, self.agent.actor.model.main.hidden_size).to(self.device)
        )
        self.hidden_cell_critic = (
            (torch.zeros(self.agent.critic.q1.LSTM.num_layers, self.num_envs, self.agent.critic.hidden_size).to(self.device),
             torch.zeros(self.agent.critic.q1.LSTM.num_layers, self.num_envs, self.agent.critic.hidden_size).to(self.device)),
            (torch.zeros(self.agent.critic.q2.LSTM.num_layers, self.num_envs, self.agent.critic.hidden_size).to(self.device),
            torch.zeros(self.agent.critic.q2.LSTM.num_layers, self.num_envs, self.agent.critic.hidden_size).to(self.device))
        )
        self.prev_action = torch.zeros(self.num_envs, 9).to(self.device)
    
    def env_step(self):
        # Plan action and take it
        state = {k: v.clone() for k, v in self.state.items()}
        action, hidden_cell_actor, hidden_cell_critic = self.agent.plan_action(state, self.hidden_cell_actor, self.hidden_cell_critic)
        next_state = self.agent.apply_action(self.state, action)
        done = next_state['remaining_strokes'] <= 0

        # Compute reward
        reward = self.agent.evaluate_state(next_state, self.state, action, self.prev_action).unsqueeze(-1)

        # Store (S, A, R, S+) in buffer
        mask = self.state['mask'].squeeze(-1).to(torch.bool)
        for i in range(self.num_envs):
            # In place for priority sampling, I implemented scaling probability to add to buffer
            # This is a bit hacky, but it works. Any reward over 50 is guaranteed add.
            # The base reward of 18-25ish has 40-50% chance of being added.
            # Also don't store masked out items
            prob = min(1.0,reward[i]/50.0)
            if mask[i] and torch.rand(1).item() < prob:
                self.buffer.push(
                    {k: v[i].detach() for k, v in self.state.items()},
                    (self.hidden_cell_actor[0][:,i].detach(),self.hidden_cell_actor[1][:,i].detach()),
                    ((self.hidden_cell_critic[0][0][:,i].detach(),self.hidden_cell_critic[0][1][:,i].detach()),(self.hidden_cell_critic[1][0][:,i].detach(),self.hidden_cell_critic[1][1][:,i].detach())),
                    action[i].detach(),
                    reward[i].detach(),
                    {k: v[i].detach() for k, v in next_state.items()},
                    done[i].detach()
                )
        
        # Update state and hidden cells
        self.state = next_state
        self.prev_action = action
        self.hidden_cell_actor = hidden_cell_actor
        self.hidden_cell_critic = hidden_cell_critic
        return reward[:,0] * mask
    
    def checkpoint(self, num, score=None):
        torch.save(self.agent.actor.model.state_dict(), os.path.join(self.save_folder,'actor_model.pth'))
        torch.save(self.agent.critic.state_dict(), os.path.join(self.save_folder,'critic.pth'))
        plt.close()
        plt.plot(self.return_history, label='Returns')
        plt.savefig(os.path.join(self.save_folder,'returns.png'))

        canvas_target = self.state['target_img'][0]
        canvas_current = self.state['current_canvas'][0]
        joint_canvas = torch.cat([canvas_target, torch.zeros((3,canvas_target.shape[1],3)).to(self.device), canvas_current], dim=-1)
        plt.close()
        plt.imshow(joint_canvas.detach().cpu().permute(1,2,0).numpy())
        title = "Target Canvas vs Current Canvas"
        if score is not None: title += f". Score: {score:.2f}"
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(self.save_folder,f"canvas_{num}.png"))

    def train(self, rollouts=1000):
        self.agent.actor_frozen = True
        for i in range(rollouts):
            with torch.no_grad():
                if i%self.generate_new_target_freq == 0:
                    self.generate_new_target = True
                self.env_reset()
                self.generate_new_target = False
                max_strokes = self.state['remaining_strokes'].max().to(torch.long).item()
                print(f"Episode {i}/{rollouts}")
                rollout_returns = torch.zeros(self.num_envs).to(self.device)
                for _ in tqdm(range(max_strokes), desc="Rolling out episode...", unit="Stroke", total=max_strokes):
                    rew = self.env_step()
                    rollout_returns = self.agent.gamma * rollout_returns + rew.squeeze(-1)
                returns = rollout_returns.mean().item()
                self.update_returns.append(returns)
                print("Avg Return: ", returns)
            
            if i>=self.seed_rollouts and self.agent.actor_frozen:
                self.agent.actor_frozen = False
            
            if i%self.update_freq == 0 and i>0:
                for _ in tqdm(range(len(self.buffer)//self.agent.batch_size), desc="Updating agent...", unit="Batch", total=len(self.buffer)//self.agent.batch_size):
                    self.agent.update(self.buffer)
                self.return_history.append(torch.tensor(self.update_returns).mean().item())
                self.checkpoint(i, score=rollout_returns[0].item())
                self.update_returns = []
                print()

        return self.agent.actor.model
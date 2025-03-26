import torch
from torch import nn
import torchvision.transforms as transforms

from actor import StrokeActor
from critic import StrokeCritic
from stroke_generator.utils.encoders import CanvasEncoder, StrokeEncoder

class StrokeAgent(nn.Module):
    def __init__(self, opt, device='cpu'):
        super().__init__()

        self.opt = opt
        self.device = device

        self.actor = StrokeActor(opt, device)
        self.critic = StrokeCritic(opt, device)
        self.target_critic = StrokeCritic(opt, device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.state_encoder = CanvasEncoder(opt, device)
        self.action_encoder = StrokeEncoder(opt, device)

        self.actor_optim = torch.optim.AdamW((self.actor.parameters(), self.state_encoder.parameters()), lr=1e-3)
        self.critic_optim = torch.optim.AdamW((self.critic.parameters(), self.action_encoder.parameters()), lr=1e-3)

        self.gamma = 0.9
        self.alpha = 0.2
        self.tau = 5e-3
    
    # state = dict{current_canvas, target_img, target_tokenized_txt, remaining_strokes, color_palette}
    # latent_state = False
    #   or
    # state = torch.Tensor of shape (batch_size, latent_dim)
    # latent_state = True
    def sample_action(self, state, latent_state=False):
        if not latent_state:
            state = self.state_encoder(**state)
        mu, log_std = self.actor(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        action = normal.rsample()
        return action

    def update(self, replay_buffer):
        state, action, reward, next_state, done = replay_buffer.sample()

        # state = dict{current_canvas, target_img, target_tokenized_txt, remaining_strokes, color_palette}
        _s = self.state_encoder(**state)
        _a = self.action_encoder(action)
        _ns = self.state_encoder(**next_state)

        # Update Critic
        with torch.no_grad():
            mu, log_std = self.actor(_s)
            std = log_std.exp()
            next_normal = torch.distributions.Normal(mu, std)
            next_action = next_normal.rsample()
            _na = self.action_encoder(next_action)

            target_q1, target_q2 = self.target_critic(_ns, _na)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_normal.log_prob(next_action).sum(dim=-1, keepdim=True)
            target_value = reward + self.gamma * (1 - done) * target_q
        
        q1, q2 = self.critic(_s, _a)
        critic_loss = nn.functional.mse_loss(q1, target_value) + nn.functional.mse_loss(q2, target_value)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Update Actor
        mu, log_std = self.actor(_s)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        action = normal.rsample()
        _a = self.action_encoder(action)

        q1, q2 = self.critic(_s, _a)
        actor_loss = (self.alpha * normal.log_prob(action).sum(dim=-1, keepdim=True) - torch.min(q1, q2)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Update Target Critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

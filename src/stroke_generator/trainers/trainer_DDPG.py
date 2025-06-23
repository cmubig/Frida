import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
import copy

from stroke_generator.model import DeterministicStrokePredictor
from stroke_generator.SAC.actor import DeterministicStrokeActor
from stroke_generator.SAC.critic import StrokeCritic
from stroke_generator.utils.replay_buffer import ReplayBuffer
from stroke_generator.env import StrokeEnv

from stroke_generator.utils.model_utils import print_memory_update

class DDPGTrainer:    
    def __init__(
        self,
        opt,
        model: DeterministicStrokePredictor,
        actor_lr=1e-4,
        critic_lr=1e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=100_000,
        batch_size=128,
        device="cpu",
        save_folder="checkpoints",
    ):
        self.device = device
        self.save_folder = save_folder

        # Actor
        self.actor = model
        self.actor_target = DeterministicStrokePredictor(opt, device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic
        state_latent_dim = self.actor.state_latent_dim
        action_latent_dim = 1024
        self.critic = StrokeCritic(opt, device, state_latent_dim, action_latent_dim)
        self.critic_target = StrokeCritic(opt, device, state_latent_dim, action_latent_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optim and buffer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Tunable params
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Misc.
        self.max_action = model.bias + model.scale / 2.0
        self.min_action = model.bias - model.scale / 2.0
        self.mse_loss = nn.MSELoss()


    def buffer_update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        hx_a_prev = self.actor.hx
        hx_c_prev = self.critic.hx

        state, hx_a, hx_c, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        # Move data to device
        state = {k:v.to(self.device) for k,v in state.items()}
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = {k:v.to(self.device) for k,v in next_state.items()}
        done = done.to(self.device)

        # Add noise to state, action, reward, and next_state
        for key in ['target_img', 'current_canvas']:
            state[key] = state[key] + 0.01 * torch.randn_like(state[key])
            next_state[key] = next_state[key] + 0.01 * torch.randn_like(next_state[key])
        action = action + 0.01 * action.mean(dim=0) * torch.randn_like(action)
        reward = reward + 0.01 * reward.mean() * torch.randn_like(reward)

        # Set hx for actor and critic
        self.actor.set_hidden_cell(hx_a, detach=True)
        self.critic.set_hidden_cell(hx_c, detach=True)
        self.actor_target.set_hidden_cell(hx_a, detach=True)
        self.critic_target.set_hidden_cell(hx_c, detach=True)

        # Critic update
        with torch.no_grad():
            next_action, _ = self.actor_target(**next_state)
            Q_1, Q_2, _ = self.evaluate_action(next_state, next_action, target=True)
            target_Q = reward + (1 - done.to(torch.float)) * self.gamma * torch.min(Q_1, Q_2)

        Q_1, Q_2, _ = self.evaluate_action(state, action, target=False)
        critic_loss = self.mse_loss(Q_1, target_Q) + self.mse_loss(Q_2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # self.check_gradients(self.critic)
        self.critic_optimizer.step()

        # Actor update
        if not self.actor_frozen:
            a, _ = self.actor(**state)
            Q_1, Q_2, _ = self.evaluate_action(state, a, target=False)
            actor_loss = -torch.min(Q_1, Q_2).mean()

            self.actor_optimizer.zero_grad()
            # self.check_gradients(self.actor)
            self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

        # Restore hx
        self.actor.set_hidden_cell(hx_a_prev, detach=False)
        self.critic.set_hidden_cell(hx_c_prev, detach=False)
    
    def first_order_update(self, reward, next_state, done, mask):
        
        # Critic Update
        detached_mask = mask.detach().clone()
        with torch.no_grad():
            self.actor_target.set_hidden_cell(self.actor.hx, detach=True)
            self.critic_target.set_hidden_cell(self.critic.hx, detach=True)
            next_action, _ = self.actor_target(**next_state)
            Qn_1, Qn_2, _ = self.evaluate_action(next_state, next_action, target=True)
            Qn = torch.min(Qn_1, Qn_2)

        target_Q = reward + (1-(done).to(torch.float)) * self.gamma * Qn
        Q_1, Q_2, hx_c = self.evaluate_action(next_state, next_action, target=False)

        target_Q = target_Q * mask
        Q_1 = Q_1 * mask
        Q_2 = Q_2 * mask
        critic_loss = self.mse_loss(Q_1, target_Q) + self.mse_loss(Q_2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True) # Save gradients for actor update
        # self.check_gradients(self.critic)
        self.critic_optimizer.step()

        # Actor update
        if self.actor_frozen:
            return hx_c
        
        masked_reward = reward*mask
        Q_1, Q_2, _ = self.evaluate_action(next_state, next_action, target=False)
        actor_loss = -(masked_reward.mean() + self.gamma*torch.min(Q_1, Q_2).mean())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # self.check_gradients(self.actor)
        self.actor_optimizer.step()

        return hx_c

        

    def _soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, env: StrokeEnv, num_episodes=1000, max_steps=1000, noise_start=0.1, noise_decay=0.995, min_noise=0.01, eval_env=None, checkpoint_interval=50):
        noise = noise_start
        episode_rewards = []
        for episode in range(num_episodes):
            self.actor_frozen = episode < 10
            env.generate_new_target = episode % 2 == 0
            state = env.reset()
            self.actor.hx = None
            self.critic.hx = (None, None)
            episode_reward = torch.zeros(env.num_envs,1)
            print(f"Starting Episode {episode+1}")
            for i in trange(env.max_strokes, desc=f"Episode {episode+1} Steps", leave=False):
                # Sample and apply action
                action, hx_a = self.select_action(state, noise)
                next_state, reward, done, mask = env.step(action, store_grads=True)
                
                hx_c = self.first_order_update(reward, next_state, done, mask)
                
                # Store (S, A, R, S+) in buffer
                self.store_transition(state, action, reward, next_state, done)

                episode_reward = (1-0.3*mask.detach().cpu())*episode_reward + \
                                 (reward*mask).detach().cpu()
                
                self.buffer_update()

                # Clean up
                env.clear_grads_and_finish_step(action, next_state)
                self.actor.set_hidden_cell(hx_a, detach=True)
                self.critic.set_hidden_cell(hx_c, detach=True)
                if done.any() and i < env.max_strokes-1:
                    # Pass in indices of done states
                    done_indices = torch.where(done)[0]
                    state = env.reset(envs=done_indices, strokes_remaining=env.max_strokes-i-1)
                    episode_reward[done_indices] = torch.zeros(len(done_indices),1)
                    self.actor.hx[0][:,done_indices] = torch.zeros_like(self.actor.hx[0][:,done_indices])
                    self.actor.hx[1][:,done_indices] = torch.zeros_like(self.actor.hx[1][:,done_indices])
                    self.critic.hx[0][0][:,done_indices] = torch.zeros_like(self.critic.hx[0][0][:,done_indices])
                    self.critic.hx[0][1][:,done_indices] = torch.zeros_like(self.critic.hx[0][1][:,done_indices])
                    self.critic.hx[1][0][:,done_indices] = torch.zeros_like(self.critic.hx[1][0][:,done_indices])
                    self.critic.hx[1][1][:,done_indices] = torch.zeros_like(self.critic.hx[1][1][:,done_indices])
                else:
                    state = env.state                
                        
            episode_rewards.append(episode_reward.mean().item())
            noise = max(noise * noise_decay, min_noise)

            if episode == 0 or (episode + 1) % checkpoint_interval == 0:
                self.save_checkpoint(episode + 1, state, episode_rewards, episode_reward)
            print(f"Episode {episode+1}, Reward: {episode_rewards[-1]:.2f}, Noise: {noise:.3f}")

            torch.cuda.empty_cache()

    def evaluate(self, env, num_episodes=5, max_steps=1000):
        total_reward = 0
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            for _ in range(max_steps):
                action = self.select_action(state, noise=0.0)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done:
                    break
            total_reward += episode_reward
        return total_reward / num_episodes

    def select_action(self, state, noise=0.0, target=False):
        if target:
            action, hidden_cell_actor = self.actor_target(**state)
        else:
            action, hidden_cell_actor = self.actor(**state)
        
        noise = torch.cat([
            (self.max_action-self.min_action)*noise*torch.randn_like(action[:,:-3]),
            torch.zeros_like(action[:,-3:])
        ], dim=-1)
        action = action + noise

        return action, hidden_cell_actor, 

    def evaluate_action(self, state, action, target=False):
        if target:
            Q_1, Q_2, hx_c = self.critic_target(
                self.actor_target.state_encoder(**state), 
                self.critic_target.stroke_encoder(action)
            )
        else:
            Q_1, Q_2, hx_c = self.critic(
                self.actor.state_encoder(**state), 
                self.critic.stroke_encoder(action)
            )
        return Q_1, Q_2, hx_c


    def store_transition(self, state, action, reward, next_state, done):
        mask = state['mask'].squeeze(-1).to(torch.bool)
        for i in range(mask.shape[0]):
            if mask[i]:
                hx_a = self.actor.hx
                if hx_a is None:
                    hx_a = (torch.zeros((self.actor.main.num_layers,
                                         action.shape[0],
                                         self.actor.main.hidden_size)).to(self.device),
                            torch.zeros((self.actor.main.num_layers,
                                         action.shape[0],
                                         self.actor.main.hidden_size)).to(self.device))
                
                hx_c = self.critic.hx
                if hx_c[0] is None:
                    hx_c = ((torch.zeros((self.critic.q1.LSTM.num_layers,
                                          action.shape[0],
                                          self.critic.q1.LSTM.hidden_size)).to(self.device),
                             torch.zeros((self.critic.q1.LSTM.num_layers,
                                          action.shape[0],
                                          self.critic.q1.LSTM.hidden_size)).to(self.device)),
                            (torch.zeros((self.critic.q2.LSTM.num_layers,
                                          action.shape[0],
                                          self.critic.q2.LSTM.hidden_size)).to(self.device),
                             torch.zeros((self.critic.q2.LSTM.num_layers,
                                          action.shape[0],
                                          self.critic.q2.LSTM.hidden_size)).to(self.device)))

                self.replay_buffer.push(
                    {k: v[i] for k, v in state.items()},
                    (hx_a[0][:,i],hx_a[1][:,i]),
                    ((hx_c[0][0][:,i],hx_c[0][1][:,i]),(hx_c[1][0][:,i],hx_c[1][1][:,i])),
                    action[i],
                    reward[i],
                    {k: v[i] for k, v in next_state.items()},
                    done[i]
                )
    
    def save_checkpoint(self, num, state, reward_history, score=None):
        checkpoint = {
            'actor_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.save_folder,"checkpoint.pth"))

        canvas_target = state['target_img'][0]
        canvas_current = state['current_canvas'][0]
        joint_canvas = torch.cat([
            canvas_target.detach().cpu(), 
            torch.zeros((3,canvas_target.shape[1],3)), 
            canvas_current.detach().cpu()
        ], dim=-1)
        plt.close()
        plt.clf()
        plt.imshow(joint_canvas.detach().cpu().permute(1,2,0).numpy())
        title = "Target Image and Current Canvas"
        if score is not None:
            title += f" (Score: {score[0].item():.2f})"
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(self.save_folder,f"canvas_{num}.png"))

        # plot history
        plt.close()
        plt.clf()
        plt.plot(reward_history)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(self.save_folder,f"reward_history.png"))
        
        plt.clf()
        plt.close('all')
    
    def check_gradients(self, model):
        # Check gradients for all parameters in the model
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"[NaN in gradient: {name}]")
                elif torch.isinf(param.grad).any():
                    print(f"[Inf in gradient: {name}]")
            else:
                print(f"[No gradient for: {name}]")
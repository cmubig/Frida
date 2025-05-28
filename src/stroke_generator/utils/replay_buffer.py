import torch 
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def push(self, state, hc_s, hc_c, action, reward, next_state, done):
        state = {k: v.cpu() for k, v in state.items()}
        hc_s = tuple(t.cpu() for t in hc_s)
        hc_c = tuple(tuple(t.cpu() for t in hc) for hc in hc_c)
        action = action.cpu()
        reward = reward.cpu()
        next_state = {k: v.cpu() for k, v in next_state.items()}
        done = done.cpu()

        self.buffer.append((state, hc_s, hc_c, action, reward, next_state, done))
    
    def sample(self, batch_size=64, device='cpu'):
        batch = random.sample(self.buffer, batch_size)
        states, hc_a, hc_c, actions, rewards, next_states, dones = zip(*batch)

        # Stack each field in the dict across the batch
        def stack_dicts(dict_list):
            return {k: torch.stack([d[k] for d in dict_list], dim=0).to(device) for k in dict_list[0]}

        states = stack_dicts(states)
        hc_a = tuple(torch.stack([t[i] for t in hc_a], dim=1).to(device) for i in range(len(hc_a[0])))
        hc_c = tuple(
            tuple(torch.stack([t[i][j] for t in hc_c], dim=1).to(device) for j in range(len(hc_c[0][0]))) for i in range(len(hc_c[0]))
        )
        next_states = stack_dicts(next_states)
        actions = torch.stack(actions).to(device)
        rewards = torch.stack(rewards).to(device)
        dones = torch.stack(dones).to(device)

        return states, hc_a, hc_c, actions, rewards, next_states, dones
    
    def clear(self):
        self.buffer.clear()

class ReplayBufferWithGroundTruth:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)
    
    def push(self, state, hc_s, hc_c, action, gt_action):
        state = {k: v.cpu() for k, v in state.items()}
        hc_s = tuple(t.cpu() for t in hc_s)
        hc_c = tuple(tuple(t.cpu() for t in hc) for hc in hc_c)
        action = action.cpu()
        gt_action = gt_action.cpu()

        self.buffer.append((state, hc_s, hc_c, action, gt_action))
    
    def sample(self, batch_size=64, device='cpu'):
        batch = random.sample(self.buffer, batch_size)
        state, hc_s, hc_c, action, gt_action = zip(*batch)

        # Stack each field in the dict across the batch
        def stack_dicts(dict_list):
            return {k: torch.stack([d[k] for d in dict_list], dim=0).to(device) for k in dict_list[0]}

        states = stack_dicts(states)
        hc_a = tuple(torch.stack([t[i] for t in hc_a], dim=1).to(device) for i in range(len(hc_a[0])))
        hc_c = tuple(
            tuple(torch.stack([t[i][j] for t in hc_c], dim=1).to(device) for j in range(len(hc_c[0][0]))) for i in range(len(hc_c[0]))
        )
        actions = torch.stack(actions).to(device)
        gt_actions = torch.stack(gt_actions).to(device)

        return states, hc_a, hc_c, actions, gt_actions
    
    def clear(self):
        self.buffer.clear()
import torch
from torch.utils.data import Dataset

class TrajDataset(Dataset):
    def __init__(self, trajectories):
        '''
        trajectories: list of [N, 3] np arrays
        '''
        self.trajectories = [torch.tensor(traj).float() for traj in trajectories]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

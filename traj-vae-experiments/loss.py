import torch

def kl_loss(mean, logvar):
    '''
    mean: (N, latent_dim)
    logvar: (N, latent_dim)
    '''
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()

def traj_mse_loss(traj1, traj2):
    '''
    traj1: (N, L, 3)
    traj2: (N, L, 3)
    '''
    position_mse = torch.mean((traj1[:,:,:2] - traj2[:,:,:2]) ** 2)
    depth_mse = torch.mean((traj1[:,:,2] - traj2[:,:,2]) ** 2)
    return position_mse + 0.2 * depth_mse

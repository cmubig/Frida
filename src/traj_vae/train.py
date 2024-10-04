# Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='model', help='model will be saved as <name>.pt')
parser.add_argument('--num_points', type=int, default=32, help='number of points in trajectory')
parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate; default is 0.001')
parser.add_argument('--beta', type=float, default=0.0001, help='coefficient for KL term of loss')
parser.add_argument('--latent_dim', type=int, default=16, help='dimension of latent space')
parser.add_argument('--pretrained', type=str, default=None, help='name of pretrained model (optional)')
parser.add_argument('--drawings', type=int, nargs='+', default=None, help='list of drawings to train from (e.g. "0 1 2"); default is all')
args = parser.parse_args()

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from traj_helpers import get_trajectories, normalize_trajectory, resample_trajectory, plot_trajectory
from traj_dataset import TrajDataset
from autoencoders import MLP_VAE
from loss import traj_mse_loss, kl_loss

if args.drawings is None:
    DRAWING_DIRS = [os.path.join("jspaint_dataset", d) for d in os.listdir("jspaint_dataset")]
else:
    DRAWING_DIRS = [f"jspaint_dataset/{i:02d}" for i in args.drawings]
LATENT_DIM = args.latent_dim
POINTS_PER_TRAJ = args.num_points
BETA = args.beta
LR = args.lr
NUM_EPOCHS = args.num_epochs
PRETRAINED_PATH = f"saved_models/{args.pretrained}.pt"
SAVE_PATH = f"saved_models/{args.name}.pt"

# Load dataset
trajectories = get_trajectories(DRAWING_DIRS)
print(f"{len(trajectories)} total trajectories in dataset")
for i in range(len(trajectories)):
    trajectories[i] = normalize_trajectory(trajectories[i])
    trajectories[i] = resample_trajectory(trajectories[i], num_points=POINTS_PER_TRAJ)

# Set up
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = TrajDataset(trajectories)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
if args.pretrained is None:
    model = MLP_VAE(
        input_points_per_traj=POINTS_PER_TRAJ,
        latent_dim=LATENT_DIM,
        output_points_per_traj=POINTS_PER_TRAJ
    )
else:
    model = torch.load(PRETRAINED_PATH)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        output, mean, logvar = model(batch)
        mse = traj_mse_loss(batch, output)
        kl = kl_loss(mean, logvar)
        loss = mse + BETA * kl
        loss.backward()
        optimizer.step()

# Save model parameters
if not os.path.exists("saved_models"):
    os.mkdir("saved_models")
torch.save(model.state_dict(), SAVE_PATH)

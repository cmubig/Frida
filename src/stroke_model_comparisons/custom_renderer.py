import torch
from torch import nn

class CustomRenderer(nn.Module):
    def __init__(self, width):
        super(CustomRenderer, self).__init__()

        self.width = width

        idxs_x = torch.arange(self.width)
        idxs_y = torch.arange(self.width)
        x_coords, y_coords = torch.meshgrid(idxs_y, idxs_x, indexing='ij') # self.width x self.width
        self.grid_coords = torch.stack((y_coords, x_coords), dim=2).reshape(1,self.width,self.width,2) # 1 x self.width x self.width x 2

        # parameters
        self.radius = nn.Parameter(torch.ones(1)*5)
        self.dark_mult = nn.Parameter(torch.ones(1)*0.5)
        self.dark_exp = nn.Parameter(torch.ones(1))
        self.dx = nn.Parameter(torch.zeros(1))
        self.dy = nn.Parameter(torch.zeros(1))

    def forward(self, traj):
        # traj: B x n x 2
        traj[:,:,0] += self.dx
        traj[:,:,1] += self.dy
        traj = traj * self.width
        B, n, _ = traj.shape

        vs = traj[:,:-1,:].reshape((B, n-1, 1, 1, 2)) # (B, n-1, 1, 1, 2)
        vs = torch.tile(vs, (1, 1, self.width, self.width, 1)) # (B, n-1, self.width, self.width, 2)

        ws = traj[:,1:,:].reshape((B, n-1, 1, 1, 2)) # (B, n-1, 1, 1, 2)
        ws = torch.tile(ws, (1, 1, self.width, self.width, 1)) # (B, n-1, self.width, self.width, 2)

        coords = torch.tile(self.grid_coords, (n-1,1,1,1)).to(ws.device) # (n-1, self.width, self.width, 2)

        # For each of the n-1 segments, compute distance from every point to the line
        def dist_line_segment(p, v, w):
            d = torch.linalg.norm(v-w, dim=4) # B x n-1 x self.width x self.width
            dot = (p-v) * (w-v) # B x n-1 x self.width x self.width x 2
            dot_sum = torch.sum(dot, dim=4) / (d**2 + 1e-5) # B x n-1 x self.width x self.width
            t = dot_sum.unsqueeze(4) # B x n-1 x self.width x self.width x 1
            t = torch.clamp(t, min=0, max=1) # N x self.width x self.width x 1
            proj = v + t * (w-v) # B x n-1 x self.width x self.width x 2
            return torch.linalg.norm(p-proj, dim=4)
        distances = dist_line_segment(coords, vs, ws) # (B, n-1, self.width, self.width)
        distances = torch.min(distances, dim=1).values # (B, self.width, self.width)

        darkness = torch.clamp((self.radius - distances) / self.radius, min=1e-8, max=1.0)
        # darkness = (darkness ** self.dark_exp) * self.dark_mult
        darkness = (darkness ** self.dark_exp)
        darkness = torch.clamp(darkness, min=0.0, max=1.0)

        return darkness

import torch
from torch import nn

from brush_stroke import BrushStrokeBatch
from painting import PaintingBatch

class Expert():
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

        self.current_canvas = None      # torch.Tensor of shape (batch_size, 3, h, w)
        self.color_palette = None       # torch.Tensor of shape (batch_size, palette_size, 3)
        self.strokes_remaining = None   # torch.Tensor of shape (batch_size, )
    
    @property
    def has_strokes_remaining(self):
        # Returns boolean mask tensor indicating if strokes are remaining for each canvas
        return (self.strokes_remaining > 0).squeeze(-1)
    
    def step(self):
        raise NotImplementedError("step() not implemented in base class")
    
    def paint_stroke(self, stroke, canvas):
        # Paint stroke[s] on canvas and save new canvas
        painting = PaintingBatch(self.opt, background_img=canvas).to(self.device)
        updated_canvas = painting([stroke], self.h_render, self.w_render, use_alpha=False, return_alphas=False)

        return updated_canvas
    
    def rollout_trajectory(self, n_strokes, current_canvas, color_palette):
        # Rolls out n_strokes randomly
        # Returns updated canvases and strokes
        #
        # Inputs:
        #     n_strokes: torch.Tensor of shape (batch_size, )
        #     current_canvas: torch.Tensor of shape (batch_size, 3, h, w)
        #
        # Outputs:
        #     updated_canvas: torch.Tensor of shape (batch_size, n_strokes, 3, h, w)
        #     stroke: torch.Tensor of shape (batch_size, n_strokes, 9)

        self.strokes_remaining = n_strokes
        self.current_canvas = current_canvas
        self.color_palette = color_palette

        batch_size = current_canvas.shape[0]
        max_strokes = int(n_strokes.max().item())

        # Shape: [batch size, stroke_horizon, 9]
        # Shape: [batch size, stroke_horizon, 3, h, w]
        strokes = -1*torch.ones(batch_size, max_strokes, 9).to(self.device)
        canvases = -1*torch.ones(batch_size, max_strokes+1, 3, self.h_render, self.w_render).to(self.device)

        canvases[:,0] = current_canvas
        for i in range(max_strokes):
            if not self.has_strokes_remaining.any():
                break

            stroke, next_canvas = self.step()
            strokes[self.has_strokes_remaining, i] = stroke
            canvases[self.has_strokes_remaining, i+1] = next_canvas
            self.strokes_remaining -= 1
        
        return strokes, canvases
    
    def clear_memory(self):
        # Clears the current canvas and color palette
        self.current_canvas = None
        self.color_palette = None
        self.strokes_remaining = None
    

class RandomExpert(Expert):
    def __init__(self, opt, device):
        super().__init__(opt, device)

    def step(self):
        # Generates a random stroke on the current canvas
        # Returns updated canvas and stroke
        #
        # Inputs:
        #     None
        #
        # Outputs:
        #     updated_canvas: torch.Tensor of shape (batch_size, 3, h, w)
        #     stroke: torch.Tensor of shape (batch_size, 9)

        current_canvas = self.current_canvas[self.has_strokes_remaining]
        batch_size = current_canvas.shape[0]

        # Generate a random brush stroke
        stroke = BrushStrokeBatch(self.opt, batch_size, ink=None, init_differentiably=False, device=self.device)

        # Randomly select a color from the color palette
        rand_color_choice = torch.randint(0, 12, (batch_size,))
        stroke.color_transform = nn.Parameter(self.color_palette[torch.arange(batch_size), rand_color_choice].to(self.device))

        # Tensorize stroke parameters
        stroke_tensor= torch.cat([
            stroke.stroke_length,
            stroke.stroke_z,
            stroke.stroke_bend,
            stroke.transformation.a,
            stroke.transformation.xt,
            stroke.transformation.yt,
            stroke.color_transform
        ], dim=1).to(self.device)

        updated_canvas = self.paint_stroke(stroke, current_canvas)
        self.current_canvas[self.has_strokes_remaining] = updated_canvas

        return stroke_tensor, updated_canvas

class GridExpert(Expert):
    def __init__(self, opt, device, grid_size=4, noise_level=0.01):
        super().__init__(opt, device)
        # range from -1 to 1
        self.grid_size = grid_size
        self.noise_level = noise_level
        linspace = torch.linspace(-0.9, 0.9, self.grid_size+2).to(self.device)[1:-1]
        self.grid = torch.stack(torch.meshgrid(linspace, linspace), dim=-1).reshape(-1, 2)
    
    def step(self):
        # Generates a random stroke on the current canvas
        # Samples XY from grid + noise
        # Returns stroke tensor and updated canvas
        #
        # Inputs:
        #     None
        #
        # Outputs:
        #     stroke_tensor: torch.Tensor of shape (batch_size, 9)
        #     updated_canvas: torch.Tensor of shape (batch_size, 3, h, w)

        current_canvas = self.current_canvas[self.has_strokes_remaining]
        batch_size = current_canvas.shape[0]

        # Generate a random brush stroke
        stroke = BrushStrokeBatch(self.opt, batch_size, ink=None, init_differentiably=False, device=self.device)

        # Randomly select a color from the color palette
        rand_color_choice = torch.randint(0, 12, (batch_size,))
        stroke.color_transform = nn.Parameter(self.color_palette[torch.arange(batch_size), rand_color_choice].to(self.device))

        # Sample XY from grid + noise
        grid_xy = self.grid[torch.randint(0, self.grid.shape[0], (batch_size,))]
        grid_xy += torch.randn_like(grid_xy) * self.noise_level
        grid_xy = torch.clamp(grid_xy, -1, 1)
        stroke.transformation.xt = nn.Parameter(grid_xy[:, 0].unsqueeze(-1))
        stroke.transformation.yt = nn.Parameter(grid_xy[:, 1].unsqueeze(-1))

        # Tensorize stroke parameters
        stroke_tensor= torch.cat([
            stroke.stroke_length,
            stroke.stroke_z,
            stroke.stroke_bend,
            stroke.transformation.a,
            stroke.transformation.xt,
            stroke.transformation.yt,
            stroke.color_transform
        ], dim=1).to(self.device)

        updated_canvas = self.paint_stroke(stroke, current_canvas)
        self.current_canvas[self.has_strokes_remaining] = updated_canvas

        return stroke_tensor, updated_canvas
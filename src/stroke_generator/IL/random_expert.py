import torch

from brush_stroke import BrushStrokeBatch
from painting import PaintingBatch

class RandomExpert():
    def __init__(self, opt):
        self.opt = opt
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

    def step(self, current_canvas):
        # Generates a random stroke on the current canvas
        # Returns updated canvas and stroke
        #
        # Inputs:
        #     current_canvas: torch.Tensor of shape (batch_size, 3, h, w)
        #
        # Outputs:
        #     updated_canvas: torch.Tensor of shape (batch_size, 3, h, w)
        #     stroke: torch.Tensor of shape (batch_size, 9)
        batch_size = current_canvas.shape[0]
        device = current_canvas.device

        stroke = BrushStrokeBatch(self.opt, batch_size, ink=None, init_differentiably=False).to(device)
        stroke_tensor= torch.cat([
            stroke.stroke_length,
            stroke.stroke_z,
            stroke.stroke_bend,
            stroke.transformation.a,
            stroke.transformation.xt,
            stroke.transformation.yt,
            stroke.color_transform
        ], dim=1)
        
        painting = PaintingBatch(self.opt, background_img=current_canvas).to(device)
        updated_canvas = painting([stroke], self.h_render, self.w_render, use_alpha=False, return_alphas=False)

        return updated_canvas, stroke_tensor

    def rollout_trajectory(self, n_strokes, current_canvas):
        # Rolls out n_strokes randomly
        # Returns updated canvases and strokes
        #
        # Inputs:
        #     n_strokes: int
        #     current_canvas: torch.Tensor of shape (batch_size, 3, h, w)
        #
        # Outputs:
        #     updated_canvas: torch.Tensor of shape (batch_size, n_strokes, 3, h, w)
        #     stroke: torch.Tensor of shape (batch_size, n_strokes, 9)

        batch_size = current_canvas.shape[0]
        device = current_canvas.device

        # Shape: [batch size, stroke_horizon, 9]
        strokes = torch.zeros(batch_size, n_strokes, 9).to(device)

        # Shape: [batch size, stroke_horizon, 3, h, w]
        canvases = torch.zeros(batch_size, n_strokes+1, 3, self.h_render, self.w_render).to(device)

        canvases[:,0] = current_canvas
        for i in range(n_strokes):
            current_canvas, stroke = self.step(current_canvas)
            strokes[:,i] = stroke
            canvases[:,i+1] = current_canvas
        
        return strokes, canvases
import torch
from torch import nn
import pydiffvg
pydiffvg.set_use_gpu(torch.cuda.is_available())

class DiffVGRenderer(nn.Module):
    def __init__(self, width):
        super(DiffVGRenderer, self).__init__()

        self.width = width
        self.radius = nn.Parameter(torch.ones(1))
        self.render = pydiffvg.RenderFunction.apply

    def forward(self, traj):
        # traj: B x n x 2
        traj = traj * self.width
        B, _, _ = traj.shape

        res = []
        for i in range(B):
            poly = pydiffvg.Polygon(
                points=traj[i],
                is_closed=False,
                stroke_width=self.radius
            )
            # poly = pydiffvg.Path(
            #     num_control_points = torch.tensor([0]),
            #     points=torch.stack([traj[i][0], traj[i][-1]], dim=0),
            #     is_closed=False
            # )
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([0]),
                fill_color=torch.tensor([0.0, 0.0, 0.0, 0.0]),
                stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0])
            )

            scene_args = pydiffvg.RenderFunction.serialize_scene(self.width, self.width, [poly], [shape_group])

            img = self.render(
                self.width, # width
                self.width, # height
                2, # num_samples_x
                2, # num_samples_y
                0, # seed
                None, # background_image
                *scene_args
            )

            res.append(img[:,:,3])

        res = torch.stack(res, dim=0)
        return res


import pydiffvg
import torch 
import random
from torch import nn

def initialize_curves(num_paths, canvas_width, canvas_height):
    shapes = []
    shape_groups = []
    for i in range(num_paths):
        num_segments = random.randint(1,1)#1, 1)#3) #################
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        p0 = (random.random(), random.random())
        points.append(p0)
        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points = num_control_points, 
                points = points, stroke_width = torch.tensor(random.random()*5.0+1.0), is_closed = False)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, 
            stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
        shape_groups.append(path_group)
    
    points_vars = []
    color_vars = []
    width_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    return shapes, shape_groups, points_vars, color_vars, width_vars

def initialize_curves_grid(canvas_width, canvas_height, target=None, num_strokes_x=25, num_strokes_y=22):
    shapes = []
    shape_groups = []

    # xys = [(x,y) for x in torch.linspace(.03,.97,num_strokes_x) for y in torch.linspace(.03,.97,num_strokes_y)]
    xys = [(x,y) for x in torch.linspace(.3,.7,num_strokes_x) for y in torch.linspace(.3,.7,num_strokes_y)]
    random.shuffle(xys)
    # print(len(xys), len(xys[0]))
    for i in range(len(xys)):
        num_segments = 1#random.randint(1,1)#1, 1)#3) #################
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        points = []
        p0 = (xys[i][0], xys[i][1])
        points.append(p0)
        for j in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            points.append(p3)
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points = num_control_points, 
                points = points, stroke_width = torch.tensor(random.random()*5.0+1.0), is_closed = False)
        shapes.append(path)

        if target is not None:
            color = target[:,:3,int(xys[i][1]*target.shape[2]), int(xys[i][0]*target.shape[3])][0]
        else:
            color = [random.random(), random.random(), random.random()]
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, 
            stroke_color = torch.tensor([color[0], color[1], color[2], 1.]))
        shape_groups.append(path_group)
    
    points_vars = []
    color_vars = []
    width_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    return shapes, shape_groups, points_vars, color_vars, width_vars

# def initialize_curves_grid(canvas_width, canvas_height, target=None, num_strokes_x=25, num_strokes_y=22):
#     shapes = []
#     shape_groups = []

#     # xys = [(x,y) for x in torch.linspace(.03,.97,num_strokes_x) for y in torch.linspace(.03,.97,num_strokes_y)]
#     xys = [(x,y) for x in torch.linspace(.3,.7,num_strokes_x) for y in torch.linspace(.3,.7,num_strokes_y)]
#     random.shuffle(xys)
#     # print(len(xys), len(xys[0]))
#     for i in range(len(xys)):
#         num_segments = 1#random.randint(1,1)#1, 1)#3) #################
#         num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 1########2
#         points = []
#         p0 = (xys[i][0], xys[i][1])
#         points.append(p0)
#         for j in range(num_segments):
#             radius = 0.1
#             p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
#             p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
#             points.append(p1)
#             points.append(p2)
#             p0 = p2
#         points = torch.tensor(points)
#         points[:, 0] *= canvas_width
#         points[:, 1] *= canvas_height
#         path = pydiffvg.Path(num_control_points = num_control_points, 
#                 points = points, stroke_width = torch.tensor(random.random()*5.0+1.0), is_closed = False)
#         shapes.append(path)

#         if target is not None:
#             color = target[:,:3,int(xys[i][1]*target.shape[2]), int(xys[i][0]*target.shape[3])][0]
#         else:
#             color = [random.random(), random.random(), random.random()]
#         path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, 
#             stroke_color = torch.tensor([color[0], color[1], color[2], 1.]))
#         shape_groups.append(path_group)
    
#     points_vars = []
#     color_vars = []
#     width_vars = []
#     for path in shapes:
#         path.points.requires_grad = True
#         points_vars.append(path.points)
#         path.stroke_width.requires_grad = True
#         width_vars.append(path.stroke_width)
#     for group in shape_groups:
#         group.stroke_color.requires_grad = True
#         color_vars.append(group.stroke_color)

#     return shapes, shape_groups, points_vars, color_vars, width_vars

def init_diffvg_brush_stroke(h, w):

    cp = torch.tensor([
        [.5*w, .5*h],
        [.5*w+.01*w, .5*h],
        [.5*w+.02*w, .5*h],
        [.5*w+.03*w, .5*h],
    ])
    shape = pydiffvg.Path(
            num_control_points=torch.tensor([2], dtype = torch.int32),
            points=cp,
            stroke_width=torch.tensor(2.),
            is_closed=False
    )
    shape_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=None, 
            stroke_color=torch.tensor([.1,.1,.1, 1.]))
    return shape, shape_group

render = pydiffvg.RenderFunction.apply
def render_drawing(shapes, shape_groups,\
                   canvas_width, canvas_height, n_iter, save=False, no_grad=False, background_img=None):
    if no_grad:
        with torch.no_grad():
            scene_args = pydiffvg.RenderFunction.serialize_scene(\
                canvas_width, canvas_height, shapes, shape_groups)
            # render = pydiffvg.RenderFunction.apply
            # render.requires_grad=False
            img = render(canvas_width, canvas_height, 2, 2, n_iter, background_img, *scene_args)
            img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])        
            # if save:
            #     pydiffvg.imwrite(img.cpu(), '/content/res/iter_{}.png'.format(int(n_iter)), gamma=1.0)
            img = img[:, :, :3]
            img = img.unsqueeze(0)
            img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
            return img
    else:
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        # print(3.1)
        
        # print(3.2)
        img = render(canvas_width, canvas_height, 2, 2, n_iter, background_img, *scene_args)
        # print(3.5)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])        
        # if save:
        #     pydiffvg.imwrite(img.cpu(), '/content/res/iter_{}.png'.format(int(n_iter)), gamma=1.0)
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        return img

def fix_shape_group_ids(shape_groups):
    with torch.no_grad():
        j = 0
        for shape_group in shape_groups:
            shape_group.shape_ids = torch.tensor([j])
            j += 1
    return shape_groups


class WidthTranslation(nn.Module):
    def __init__(self):
        super(WidthTranslation, self).__init__()
        # self.a = nn.Parameter(torch.tensor(3./1024.))
        # #self.e = nn.Parameter(torch.tensor(1.))
        # self.b = nn.Parameter(torch.tensor(4./1024.))

        self.a = nn.Parameter(torch.tensor(0.01))
        self.b = nn.Parameter(torch.tensor(0.0036))

    def forward(self, real_widths):
        return self.real_to_sim(real_widths)

    def real_to_sim(self, real_widths):
        #return torch.mean(real_widths)**self.e*self.a + self.b
        return torch.mean(real_widths)*self.a + self.b
    def sim_to_real(self, sim_width): # Sim width should be proportionate to the height of the drawing
        #return ((sim_width - self.b)/self.a)**(1/self.e)
        return (sim_width - self.b)/self.a

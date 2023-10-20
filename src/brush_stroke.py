import copy
import math
import torch
from torch import nn
import torchgeometry
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode 
bicubic = InterpolationMode.BICUBIC
import warnings
import numpy as np

from param2stroke import special_sigmoid


def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.
        https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
    Input
        :param roll: The roll (rotation around x-axis) angle in radians.
        :param pitch: The pitch (rotation around y-axis) angle in radians.
        :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
        :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    # roll = (roll/360.)*2.*math.pi
    # pitch = (pitch/360.)*2.*math.pi
    # yaw = (yaw/360.)*2.*math.pi
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def spherical_to_quaternion(theta, phi):
    return np.cos(theta)*np.sin(phi), \
            np.sin(theta)*np.sin(phi), \
            np.cos(phi)

cos = torch.cos
sin = torch.sin
def rigid_body_transform(a, xt, yt, anchor_x, anchor_y):
    # a is the angle in radians, xt and yt are translation terms of pixels
    # anchor points are where to rotate around (usually the center of the image)
    # Blessed be Peter Schorn for the anchor point transform https://stackoverflow.com/a/71405577
    A = torch.zeros(1, 3, 3).to(a.device)
    a = -1.*a
    A[0,0,0] = cos(a)
    A[0,0,1] = -sin(a)
    A[0,0,2] = anchor_x - anchor_x * cos(a) + anchor_y * sin(a) + xt#xt
    A[0,1,0] = sin(a)
    A[0,1,1] = cos(a)
    A[0,1,2] = anchor_y - anchor_x * sin(a) - anchor_y * cos(a) + yt#yt
    A[0,2,0] = 0
    A[0,2,1] = 0
    A[0,2,2] = 1
    return A

class RigidBodyTransformation(nn.Module):
    def __init__(self, a, xt, yt):
        super(RigidBodyTransformation, self).__init__()
        # weights = torch.zeros(3)
        # weights[0] = a
        # weights[1] = xt
        # weights[2] = yt
        # weights.requires_grad = True
        # self.weights = nn.Parameter(weights)

        # t = torch.ones(1)
        # t[0] = xt
        # t.requires_grad = True
        # self.xt = nn.Parameter(t)

        # t = torch.ones(1)
        # t[0] = yt
        # t.requires_grad = True
        # self.yt = nn.Parameter(t)


        # t = torch.ones(1)
        # t[0] = a
        # t.requires_grad = True
        # self.a = nn.Parameter(t)

        # self.xt.requires_grad = True
        # self.yt.requires_grad = True
        # self.a.requires_grad = True

        self.xt = nn.Parameter(torch.ones(1)*xt)
        self.yt = nn.Parameter(torch.ones(1)*yt)
        self.a = nn.Parameter(torch.ones(1)*a)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        anchor_x, anchor_y = w/2, h/2

        # M = rigid_body_transform(self.weights[0], self.weights[1]*(w/2), self.weights[2]*(h/2), anchor_x, anchor_y)
        M = rigid_body_transform(self.a[0], self.xt[0]*(w/2), self.yt[0]*(h/2), anchor_x, anchor_y)
        with warnings.catch_warnings(): # suppress annoing torchgeometry warning
            warnings.simplefilter("ignore")
            return torchgeometry.warp_perspective(x, M, dsize=(h,w))

class BrushStroke(nn.Module):
    def __init__(self, 
                 opt,
                #stroke_length=None, stroke_z=None, stroke_bend=None, stroke_alpha=None,
                path=None,
                ctrl_pts=4,
                color=None, 
                ink=False,
                a=None, xt=None, yt=None,
                device='cuda'):
        super(BrushStroke, self).__init__()

        self.MAX_STROKE_LENGTH = opt.MAX_STROKE_LENGTH
        self.MIN_STROKE_LENGTH = opt.MIN_STROKE_LENGTH
        self.MIN_STROKE_Z = opt.MIN_STROKE_Z
        self.MAX_ALPHA = opt.MAX_ALPHA
        self.MAX_BEND = opt.MAX_BEND

        if color is None: color=(torch.rand(3).to(device)*.4)+0.3
        if a is None: a=(torch.rand(1)*2-1)*3.14
        if xt is None: xt=(torch.rand(1)*2-1)
        if yt is None: yt=(torch.rand(1)*2-1)


        # if stroke_length is None: stroke_length=torch.rand(1)*self.MAX_STROKE_LENGTH
        # if stroke_z is None: stroke_z = torch.rand(1).clamp(self.MIN_STROKE_Z, 0.95)
        # if stroke_alpha is None: stroke_alpha=(torch.rand(1)*2-1)*self.MAX_ALPHA
        # if stroke_bend is None: stroke_bend = (torch.rand(1)*2 - 1) * self.MAX_BEND
        # stroke_bend = min(stroke_bend, stroke_length) if stroke_bend > 0 else max(stroke_bend, -1*stroke_length)

        self.transformation = RigidBodyTransformation(a, xt, yt)
        
        # self.stroke_length = stroke_length
        # self.stroke_z = stroke_z
        # self.stroke_bend = stroke_bend
        # self.stroke_alpha = stroke_alpha

        # self.stroke_length.requires_grad = True
        # self.stroke_z.requires_grad = True
        # self.stroke_bend.requires_grad = True
        # self.stroke_alpha.requires_grad = True

        # self.stroke_length = nn.Parameter(self.stroke_length)
        # self.stroke_z = nn.Parameter(self.stroke_z)
        # self.stroke_bend = nn.Parameter(self.stroke_bend)
        # self.stroke_alpha = nn.Parameter(self.stroke_alpha)

        if path is None: 
            path = torch.rand((ctrl_pts,4))
            path[:,0] *= self.MAX_STROKE_LENGTH
            path[:,1] = (path[:,1]*2 - 1) * self.MAX_BEND
            path[:,2] = torch.clamp(path[:,0], self.MIN_STROKE_Z, 0.95)
            path[:,3] *= self.MAX_ALPHA

            path[0,0] = 0
            path[0,1] = 0
            path[-1,1] = 0
        self.path = path 
        self.path.requires_grad = True 
        self.path = nn.Parameter(self.path)

        if not ink:
            self.color_transform = nn.Parameter(color)
        else:
            self.color_transform = torch.zeros(3).to(device)

    def forward(self, h, w, param2img):
        # # Do rigid body transformation
        full_param = self.path.unsqueeze(0)
        stroke = param2img(full_param, h, w).unsqueeze(0)

        # Pad 1 or two to make it fit
        if stroke.shape[2] != h or stroke.shape[3] != w:
            stroke = T.Resize((h, w), bicubic)(stroke)

        # from paint_utils3 import show_img
        # show_img(stroke)
        x = self.transformation(stroke)
        # show_img(x)

        # Remove stray color from the neural network being sloppy
        x = special_sigmoid(x)
        # import kornia as K
        # x = K.filters.median_blur(x, (3,3))
        
        # show_img(x)
        x = torch.cat([x,x,x,x], dim=1)

        # Color change
        x = torch.cat((x[:,:3]*0 + self.color_transform[None,:,None,None], x[:,3:]), dim=1)
        return x

    def make_valid(stroke):
        with torch.no_grad():
            stroke.path[:,0].data.clamp_(stroke.MIN_STROKE_LENGTH, stroke.MAX_STROKE_LENGTH)
            stroke.path[:,1].data.clamp_(-1.0*stroke.MAX_BEND, stroke.MAX_BEND)
            stroke.path[:,2].data.clamp_(stroke.MIN_STROKE_Z, 0.95)
            stroke.path[:,3].data.clamp_(-1.0*stroke.MAX_ALPHA, stroke.MAX_ALPHA)

            stroke.path[0,:2] = 0
            stroke.path[-1,1] = 0

            stroke.transformation.xt.data.clamp_(-1.,1.)
            stroke.transformation.yt.data.clamp_(-1.,1.)

            #stroke.color_transform.data.clamp_(0.02,0.75)
            if stroke.color_transform.min() < 0.35:
                # If it's a colored stroke, don't let it go to a flourescent color
                stroke.color_transform.data.clamp_(0.02,0.70)
            else:
                # Well balanced RGB, less constraint
                stroke.color_transform.data.clamp_(0.02,0.85)
    
    def simple_parameterization_to_bezier_points(stroke_length, bend, z, alpha=0):
        xs = (np.arange(4)/3.) * stroke_length

        trajectory=[
            [xs[0], 0, .2, alpha],
            [xs[1], bend, z, alpha],
            [xs[2], bend, z, alpha],
            [xs[3], 0, .2, alpha],
        ]
        return trajectory

    def execute(self, painter, x_start, y_start, rotation, 
                step_size=.005, curve_angle_is_rotation=False):
        # x_start, y_start in global coordinates. rotation in radians
        # curve_angle_is_rotation if true, then the brush is angled constantly down towards theta
        smooth = True
        if smooth:
            all_positions = []
            all_orientations = []

        # Need to translate x,y a bit to be accurate according to camera
        if painter.H_coord is not None:
            # Translate the coordinates so they're similar. see coordinate_calibration
            sim_coords = np.array([x_start, y_start, 1.])
            real_coords = painter.H_coord.dot(sim_coords)
            x_start, y_start = real_coords[0]/real_coords[2], real_coords[1]/real_coords[2]

        z_range = np.abs(painter.Z_MAX_CANVAS - painter.Z_CANVAS)

        from scipy.interpolate import make_interp_spline
        path = self.path.detach().cpu().numpy()
        t = range(0, len(path))

        b_x = make_interp_spline(t,path[:,0])
        b_y = make_interp_spline(t,path[:,1])
        b_z = make_interp_spline(t,path[:,2])
        b_alpha = make_interp_spline(t,path[:,3])

        approx_len = 0.0
        for i in range(len(path)-1):
            approx_len += ((path[i,0]-path[i+1,0])**2 + (path[i,1] - path[i+1,1])**2)**0.5
        # print('approx_len', approx_len)
        steps = int(max(3, approx_len/step_size))
        
        t_new = np.linspace(0, len(path)-1, steps)
        x_new = b_x(t_new)
        y_new = b_y(t_new)
        z_new = b_z(t_new)
        alpha_new = b_alpha(t_new)

        path = np.stack((x_new, y_new, z_new, alpha_new)).T

        path = BrushStroke.get_rotated_trajectory(rotation, path)

        painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.03, speed=0.4)
        painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.005, speed=0.1)

        for step in range(steps):
            x, y, z, alpha = path[step,0], path[step,1], path[step,2], path[step,3]
            x_next = x_start + x 
            y_next = y_start + y
            z = painter.Z_CANVAS - z * z_range
            q = None # TODO incorporate alpha
            

            # If off the canvas, lift up
            if (x_next > painter.opt.X_CANVAS_MAX) or (x_next < painter.opt.X_CANVAS_MIN) or \
                    (y_next > painter.opt.Y_CANVAS_MAX) or (y_next < painter.opt.Y_CANVAS_MIN):
                z += 0.005

            # Don't over shoot the canvas
            x_next = min(max(painter.opt.X_CANVAS_MIN, x_next), painter.opt.X_CANVAS_MAX) 
            y_next = min(max(painter.opt.Y_CANVAS_MIN, y_next), painter.opt.Y_CANVAS_MAX)

            if smooth:
                if t == 0 and i==0:
                    all_positions.append([x_next, y_next, z+0.02])
                    all_orientations.append(q)
                    all_positions.append([x_next, y_next, z+0.005])
                    all_orientations.append(q)
                all_positions.append([x_next, y_next, z])
                all_orientations.append(q)
                if t == 1 and (i == len(path)-4):
                    all_positions.append([x_next, y_next, z+0.01])
                    all_orientations.append(q)
                    all_positions.append([x_next, y_next, z+0.02])
                    all_orientations.append(q)
            else:
                if t == 0 and i==0:
                    painter.move_to(x_next, y_next, z+0.02, q=q, method='direct', speed=0.1)
                    painter.move_to(x_next, y_next, z+0.005, q=q, method='direct', speed=0.03)
                painter.move_to(x_next, y_next, z, q=q, method='direct', speed=0.05)
                if t == 1 and (i == len(path)-4):
                    painter.move_to(x_next, y_next, z+0.01, q=q, method='direct', speed=0.03)
                    painter.move_to(x_next, y_next, z+0.02, q=q, method='direct', speed=0.1)


        if smooth:
            stroke_complete = painter.move_to_trajectories(all_positions, all_orientations)
        

        # Don't over shoot the canvas
        x_next = x_start+path[-1,0]
        y_next = y_start+path[-1,1]
        x_next = min(max(painter.opt.X_CANVAS_MIN, x_next), painter.opt.X_CANVAS_MAX) 
        y_next = min(max(painter.opt.Y_CANVAS_MIN, y_next), painter.opt.Y_CANVAS_MAX)
        painter.move_to(x_next, y_next, painter.Z_CANVAS + 0.04, speed=0.3)
        # painter.hover_above(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS)

        return stroke_complete
    
    def get_rotated_trajectory(rotation, trajectory):
        # Rotation in radians
        ret = copy.deepcopy(trajectory)
        for i in range(len(ret)):
            ret[i][0] = math.cos(rotation) * trajectory[i][0] \
                     - math.sin(rotation) * trajectory[i][1]
            ret[i][1] = math.sin(rotation) * trajectory[i][0] \
                     + math.cos(rotation) * trajectory[i][1]
        ret = np.array(ret)
        return ret
    
    def dot_stroke(self, opt):
        path = torch.zeros((4,4))
        path[:,2] = 0.75
        return BrushStroke(
            opt,
            path=path,
        )
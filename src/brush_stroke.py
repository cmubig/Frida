import copy
import math
import torch
from torch import nn
import torchgeometry
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode 
from kornia.geometry.transform import rotate
bicubic = InterpolationMode.BICUBIC
import warnings
import numpy as np
from scipy.interpolate import make_interp_spline
from traj_vae.autoencoders import MLP_VAE

# from param2stroke import special_sigmoid


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

def rigid_body_transforms(a, xt, yt, anchor_x, anchor_y):
    # a is the angle in radians, xt and yt are translation terms of pixels
    # anchor points are where to rotate around (usually the center of the image)
    # Blessed be Peter Schorn for the anchor point transform https://stackoverflow.com/a/71405577
    A = torch.zeros(len(a), 3, 3).to(a.device)
    a = -1.*a
    A[:,0,0] = cos(a)
    A[:,0,1] = -sin(a)
    A[:,0,2] = anchor_x - anchor_x * cos(a) + anchor_y * sin(a) + xt#xt
    A[:,1,0] = sin(a)
    A[:,1,1] = cos(a)
    A[:,1,2] = anchor_y - anchor_x * sin(a) - anchor_y * cos(a) + yt#yt
    A[:,2,0] = 0
    A[:,2,1] = 0
    A[:,2,2] = 1
    return A

def get_rotation_transform(a, anchor_x, anchor_y):
    # a is the angle in radians
    # anchor points are where to rotate around (usually the center of the image)
    # Blessed be Peter Schorn for the anchor point transform https://stackoverflow.com/a/71405577
    A = torch.zeros(1, 3, 3).to(a.device)
    a = -1.*a
    A[0,0,0] = cos(a)
    A[0,0,1] = -sin(a)
    A[0,0,2] = anchor_x - anchor_x * cos(a) + anchor_y * sin(a) 
    A[0,1,0] = sin(a)
    A[0,1,1] = cos(a)
    A[0,1,2] = anchor_y - anchor_x * sin(a) - anchor_y * cos(a) 
    A[0,2,0] = 0
    A[0,2,1] = 0
    A[0,2,2] = 1
    return A

def get_translation_transform(xt, yt):
    A = torch.zeros(1, 3, 3).to(xt.device)
    A[0,0,0] = 1
    A[0,0,1] = 0
    A[0,0,2] = xt
    A[0,1,0] = 0
    A[0,1,1] = 1
    A[0,1,2] = yt
    A[0,2,0] = 0
    A[0,2,1] = 0
    A[0,2,2] = 1
    return A

class BrushStroke(nn.Module):
    vae = MLP_VAE(32, 16, 32)
    vae.load_state_dict(torch.load("traj_vae/saved_models/model.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Bounds for the x/y coordinates the VAE can generate
    # These values are calculated based on the imposed max stroke length of 0.25 in the JSPaint webapp,
    # in conjunction with the fact that the stroke starts at (0,0) and is rotated to be horizontal
    vae_max_stroke_length = 0.25
    vae_x_bounds = (-vae_max_stroke_length/2, vae_max_stroke_length)
    vae_y_bounds = (-vae_max_stroke_length/2, vae_max_stroke_length/2)

    def __init__(self, 
                 opt,
                latent=None,
                color=None, 
                ink=False,
                a=None, xt=None, yt=None,
                device='cuda',
                is_dot=False):
        super(BrushStroke, self).__init__()

        self.is_dot = is_dot

        self.MAX_STROKE_LENGTH = opt.MAX_STROKE_LENGTH
        self.MIN_STROKE_Z = opt.MIN_STROKE_Z
        self.max_length_before_new_paint = opt.max_length_before_new_paint
        self.ink = ink

        if color is None: color=(torch.rand(3).to(device)*.4)+0.3
        if a is None: a=(torch.rand(1)*2-1)*3.14#torch.zeros(1)#(torch.rand(1)*2-1)*3.14#torch.zeros(1)#
        if xt is None: xt=torch.rand(1)
        if yt is None: yt=torch.rand(1)

        self.xt = nn.Parameter(torch.ones(1)*xt) # Range [0,1]
        self.yt = nn.Parameter(torch.ones(1)*yt) # Range [0,1]
        self.a = nn.Parameter(torch.ones(1)*a) # Range [-2pi,2pi]
        
        if latent is None: 
            latent = torch.randn(1, BrushStroke.vae.latent_dim)

        self.latent = latent 
        self.latent.requires_grad = True 
        self.latent = nn.Parameter(self.latent)

        if not self.ink:
            self.color_transform = nn.Parameter(color)
        else:
            self.color_transform = torch.zeros(3).to(device)

    def forward(self, h, w, param2img, use_conv=True):
        # Do rigid body transformation
        full_param = self.get_path(normalize=False).unsqueeze(0) # 1 x 32 x 3
        
        stroke = param2img(full_param, self.xt, self.yt, self.a, use_conv=use_conv).unsqueeze(0)

        # Pad 1 or two to make it fit
        if stroke.shape[2] != h or stroke.shape[3] != w:
            stroke = T.Resize((h, w), bicubic, antialias=True)(stroke)

        # from paint_utils3 import show_img
        # show_img(stroke)
        x = stroke
        # show_img(x)

        # Remove stray color from the neural network being sloppy
        # x = special_sigmoid(x)
        # import kornia as K
        # x = K.filters.median_blur(x, (3,3))
        
        # show_img(x)
        x = torch.cat([x,x,x,x], dim=1)

        # Color change
        x = torch.cat((x[:,:3]*0 + self.color_transform[None,:,None,None], x[:,3:]), dim=1)
        return x

    def make_valid(stroke):
        with torch.no_grad():
            stroke.latent.data.clamp(-2.5, 2.5)

            stroke.xt.data.clamp_(0.05,0.95)
            stroke.yt.data.clamp_(0.05,0.95)

            #stroke.color_transform.data.clamp_(0.02,0.75)
            if stroke.color_transform.min() < 0.35:
                # If it's a colored stroke, don't let it go to a flourescent color
                stroke.color_transform.data.clamp_(0.02,0.70)
            else:
                # Well balanced RGB, less constraint
                stroke.color_transform.data.clamp_(0.02,0.85)

    def get_path(self, normalize=True):
        if self.is_dot:
            path = torch.zeros((4,4))
            path[:,2] = 0.75
            return path
        BrushStroke.vae.to(self.latent.device)
        path = BrushStroke.vae.decode(self.latent)

        # Clone the path so that the operation is not in-place (PyTorch quirk; allows gradients to flow through)
        path_clone = path.clone()
        if normalize:
            path_clone[:,0:2] = path[:,0:2] / BrushStroke.vae_max_stroke_length * self.MAX_STROKE_LENGTH
        path_clone[:,2] = path[:,2].clamp(self.MIN_STROKE_Z, 0.95)
        
        return path_clone

    def execute(self, painter, x_start, y_start, rotation, fast=False):
        # x_start, y_start in global coordinates. rotation in radians
        # curve_angle_is_rotation if true, then the brush is angled constantly down towards theta
        all_positions = []
        all_orientations = []

        # Need to translate x,y a bit to be accurate according to camera
        if painter.H_coord is not None:
            # Translate the coordinates so they're similar. see coordinate_calibration
            sim_coords = np.array([x_start, y_start, 1.])
            real_coords = painter.H_coord.dot(sim_coords)
            x_start, y_start = real_coords[0]/real_coords[2], real_coords[1]/real_coords[2]

        z_range = np.abs(painter.Z_MAX_CANVAS - painter.Z_CANVAS)

        path = self.get_path()

        approx_len = 0.0
        for i in range(len(path)-1):
            approx_len += ((path[i,0]-path[i+1,0])**2 + (path[i,1] - path[i+1,1])**2)**0.5

        path = BrushStroke.get_rotated_trajectory(rotation, path)

        # painter.move_to(x_start+path[0,0], y_start+path[0,1], painter.Z_CANVAS + 0.03, speed=0.4)

        for step in range(len(path)):
            x, y, z = path[step,0], path[step,1], path[step,2]
            x_next = x_start + x 
            y_next = y_start + y
            z = painter.Z_CANVAS - z * z_range
            q = None

            # If off the canvas, lift up
            if (x_next > painter.opt.X_CANVAS_MAX) or (x_next < painter.opt.X_CANVAS_MIN) or \
                    (y_next > painter.opt.Y_CANVAS_MAX) or (y_next < painter.opt.Y_CANVAS_MIN):
                z += 0.005

            # Don't over shoot the canvas
            x_next = min(max(painter.opt.X_CANVAS_MIN, x_next), painter.opt.X_CANVAS_MAX) 
            y_next = min(max(painter.opt.Y_CANVAS_MIN, y_next), painter.opt.Y_CANVAS_MAX)

            if step == 0: # First point: lower pen down
                all_positions.append([x_next, y_next, z+0.03])
                all_orientations.append(q)
                all_positions.append([x_next, y_next, z+0.005])
                all_orientations.append(q)
            all_positions.append([x_next, y_next, z]) # Move to point
            all_orientations.append(q)
            if step == len(path)-1: # Last point: raise pen up
                all_positions.append([x_next, y_next, z+0.01])
                all_orientations.append(q)
                all_positions.append([x_next, y_next, z+0.04])
                all_orientations.append(q)

        # Remove redundant positions (doesn't matter whether we do it or not)
        def remove_redundant_positions(positions, orientations, thresh=0.00025):
            new_positions, new_orientations = [positions[0]], [orientations[0]]
            prev_pos = np.array(positions[0])
            for i in range(1, len(positions)-1, 2):
                p1, p2, p3 = prev_pos, np.array(positions[i]), np.array(positions[i+1])
                # https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points

                delta = np.linalg.norm(np.cross(p3-p1, p1-p2))/np.linalg.norm(p3-p1)
                delta = np.nan_to_num(delta)
                # print(delta)
                if delta > thresh:
                    new_positions.append(positions[i])
                    new_orientations.append(orientations[i])
                new_positions.append(positions[i+1])
                new_orientations.append(orientations[i+1])
                prev_pos = np.array(positions[i+1])
            new_positions.append(positions[-1])
            new_orientations.append(orientations[-1])
            return new_positions, new_orientations
        
        if not self.is_dot:
            prev_n = len(all_positions)
            all_positions, all_orientations = remove_redundant_positions(all_positions, all_orientations)
            removed_positions = prev_n - len(all_positions)
            print('removed positions1', removed_positions)
            
            prev_n = len(all_positions)
            all_positions, all_orientations = remove_redundant_positions(all_positions, all_orientations)
            removed_positions = prev_n - len(all_positions)
            print('removed positions2', removed_positions)
            
            prev_n = len(all_positions)
            all_positions, all_orientations = remove_redundant_positions(all_positions, all_orientations)
            removed_positions = prev_n - len(all_positions)
            print('removed positions3', removed_positions)
        
        stroke_complete = painter.move_to_trajectories(all_positions, all_orientations, fast=fast)

        # Don't over shoot the canvas
        x_next = x_start+path[-1,0]
        y_next = y_start+path[-1,1]
        x_next = min(max(painter.opt.X_CANVAS_MIN, x_next), painter.opt.X_CANVAS_MAX) 
        y_next = min(max(painter.opt.Y_CANVAS_MIN, y_next), painter.opt.Y_CANVAS_MAX)
        # painter.move_to(x_next, y_next, painter.Z_CANVAS + 0.04, speed=0.3)
        # painter.hover_above(x_start+path[-1,0], y_start+path[-1,1], painter.Z_CANVAS)

        return stroke_complete
    
    def get_length(self):
        """ Return the approximate length of the stroke in distance (meters) """
        path = self.get_path()
        xs = path[:,0]
        ys = path[:,1]
        approx_len = (((xs[1:] - xs[:-1])**2 + (ys[1:] - ys[:-1])**2)**0.5).sum()
        return approx_len
    
    def get_rotated_trajectory(rotation, trajectory):
        # Rotation in radians
        ret = trajectory.detach().clone()
        for i in range(len(ret)):
            ret[i][0] = math.cos(rotation) * trajectory[i][0] \
                     - math.sin(rotation) * trajectory[i][1]
            ret[i][1] = math.sin(rotation) * trajectory[i][0] \
                     + math.cos(rotation) * trajectory[i][1]
        ret = ret.cpu().detach().numpy()
        return ret
    
    def dot_stroke(self, opt):
        return BrushStroke(
            opt,
            is_dot=True
        )
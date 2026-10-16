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
from filleted_stroke import radii_from_kappa, sample_filleted_polyline

# from param2stroke import special_sigmoid

N_VERTICES = 4  # polyline corners; N-2 = 2 fillet κ values
N_PATH_POINTS = 32  # dense trajectory length expected by param2img / execute


def _logit(p: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


# Shape families for stroke-library collection (Option A).
LIBRARY_SHAPE_FAMILIES = (
    'straight',   # near-linear baseline marks
    'arc',        # gentle C-curve
    's_curve',    # inflection
    'corner',     # angled polyline, sharp κ
    'hook',       # bend near the end
    'round_turn', # soft C-curve with round κ
)


def _opt_float(opt, name, default):
    '''Read a float from Options/materials without broken getattr defaults.'''
    try:
        val = float(getattr(opt, name))
        if val == val:  # not NaN
            return val
    except Exception:
        pass
    return float(default)


def _sample_library_length(opt) -> float:
    '''Stratified short / medium / long in [MIN_STROKE_LENGTH, MAX_STROKE_LENGTH].'''
    lo = _opt_float(opt, 'MIN_STROKE_LENGTH', 0.0025) or 0.0025
    hi = _opt_float(opt, 'MAX_STROKE_LENGTH', 0.1) or 0.1
    hi = max(hi, lo * 1.01)
    # Equal mass on three length bands (log-uniform within band).
    band = int(torch.randint(0, 3, (1,)).item())
    edges = torch.log(torch.tensor([lo, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi]))
    t = torch.rand(1).item()
    log_len = edges[band] + t * (edges[band + 1] - edges[band])
    return float(torch.exp(log_len).item())


def _library_vertices_and_kappa(family: str, length: float, device):
    '''Build (N_VERTICES, 2) vertices and (N_VERTICES-2,) κ from length-relative shape amps.

    Lateral size is a fraction of stroke length (can be large vs length for
    loopy / bent marks). Absolute amp is clamped near make_valid (±0.15 m).
    '''
    L = max(length, 1e-4)
    amp_abs_cap = 0.14  # meters; safety vs make_valid clamp (±0.15)
    t = torch.linspace(0.0, 1.0, N_VERTICES, device=device)
    sign = 1.0 if torch.rand(1).item() < 0.5 else -1.0

    def _u(lo, hi):
        return lo + (hi - lo) * torch.rand(1).item()

    def _cap(amp):
        return min(float(amp), amp_abs_cap)

    if family == 'straight':
        # Mostly straight, but allow a visible wiggle on longer marks.
        amp = _cap(_u(0.02, 0.12) * L)
        xs = t * L
        ys = sign * amp * (torch.rand(N_VERTICES, device=device) - 0.5) * 2.0
        kappa = torch.empty(N_VERTICES - 2, device=device).uniform_(0.10, 0.55)

    elif family == 'arc':
        # Strong C-curves; optionally chord-compress x so the bulge isn't just a tall sine.
        amp = _cap(_u(0.20, 0.65) * L)
        chord = _u(0.55, 1.0)  # <1 pulls endpoints closer → rounder arc
        xs = t * (L * chord)
        ys = sign * amp * torch.sin(math.pi * t)
        kappa = torch.empty(N_VERTICES - 2, device=device).uniform_(0.30, 0.85)

    elif family == 's_curve':
        amp = _cap(_u(0.18, 0.55) * L)
        xs = t * L
        # Uneven phase so the inflection isn't always centered.
        phase = _u(-0.15, 0.15)
        ys = sign * amp * torch.sin(2.0 * math.pi * (t + phase))
        kappa = torch.empty(N_VERTICES - 2, device=device).uniform_(0.25, 0.80)

    elif family == 'corner':
        # Sharp bend; second leg can be long and steep (not just a small dogleg).
        lat = _cap(_u(0.25, 0.85) * L)
        x_bend = _u(0.25, 0.65) * L
        leg2 = _u(0.25, 0.90) * L  # length of the second leg along x
        xs = torch.tensor([0.0, x_bend, x_bend + 0.05 * L, x_bend + leg2], device=device)
        ys = torch.tensor([0.0, 0.0, sign * lat, sign * lat * _u(0.7, 1.15)], device=device)
        ys = ys + (torch.rand(N_VERTICES, device=device) - 0.5) * 0.20 * lat
        kappa = torch.empty(N_VERTICES - 2, device=device).uniform_(0.02, 0.28)

    elif family == 'hook':
        # Late bend that can curl back a bit (x not strictly increasing).
        amp = _cap(_u(0.25, 0.70) * L)
        xs = t * L
        envelope = torch.clamp((t - _u(0.20, 0.45)) / 0.55, 0.0, 1.0) ** 2
        ys = sign * amp * envelope
        # Occasional curl: pull the tip back toward the start in x.
        if torch.rand(1).item() < 0.45:
            xs = xs.clone()
            xs[-1] = xs[-2] - _u(0.05, 0.35) * L
            xs[-2] = xs[-2] - _u(0.0, 0.10) * L
        kappa = torch.empty(N_VERTICES - 2, device=device).uniform_(0.15, 0.75)

    elif family == 'round_turn':
        # Soft U / C with large amplitude and high fillet κ.
        amp = _cap(_u(0.25, 0.70) * L)
        chord = _u(0.45, 0.95)
        xs = t * (L * chord)
        ys = sign * amp * torch.sin(math.pi * t)
        kappa = torch.empty(N_VERTICES - 2, device=device).uniform_(0.70, 0.98)

    else:
        raise ValueError(f'Unknown library shape family: {family}')

    # Small natural jitter so marks aren't perfectly synthetic.
    vertices = torch.stack([xs, ys], dim=-1)
    vertices = vertices + 0.015 * L * (torch.rand_like(vertices) - 0.5)
    vertices[0] = 0.0  # keep start anchored before recentering below
    # Anchor at origin for predictable packing / execute start.
    vertices = vertices - vertices[0:1]
    return vertices, kappa


def _sample_library_z(device, min_z=0.1) -> torch.Tensor:
    '''Sample pressure across the usable tip-calib range (0 light → 1 deepest).

    Mixes flat strokes with strong within-stroke profiles (taper, swell, random)
    so the stroke model sees real pressure → width changes.
    '''
    lo, hi = float(min_z), 0.95
    span = hi - lo
    u = torch.rand(1, device=device).item()

    if u < 0.25:
        # Nearly constant pressure (still covers the full range across strokes)
        z = lo + span * torch.rand(1, device=device).expand(4)
    elif u < 0.50:
        # Taper light → heavy or heavy → light
        a, b = lo + span * torch.rand(2, device=device)
        t = torch.linspace(0, 1, 4, device=device)
        z = a + (b - a) * t
    elif u < 0.75:
        # Swell or pinch in the middle
        ends = lo + span * torch.rand(1, device=device)
        mid = lo + span * torch.rand(1, device=device)
        z = torch.stack([ends[0], mid[0], mid[0], ends[0]])
        # Slight asymmetry so it isn't perfectly symmetric
        z = z + 0.05 * span * (torch.rand(4, device=device) - 0.5)
    else:
        # Independent control points — can jump a lot along the stroke
        z = lo + span * torch.rand(4, device=device)

    return z.clamp(lo, hi)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, 
                 opt,
                vertices=None,
                radii=None,
                color=None, 
                ink=False,
                a=None, xt=None, yt=None, 
                z=None,
                device=None,
                is_dot=False):
        super(BrushStroke, self).__init__()

        self.is_dot = is_dot

        self.MAX_STROKE_LENGTH = opt.MAX_STROKE_LENGTH
        self.MIN_STROKE_Z = opt.MIN_STROKE_Z
        self.max_length_before_new_paint = opt.max_length_before_new_paint
        self.ink = ink

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if color is None: color=(torch.rand(3, device=device)*.4)+0.3
        if a is None: a=(torch.rand(1, device=device)*2-1)*3.14
        if xt is None: xt=torch.rand(1, device=device)
        if yt is None: yt=torch.rand(1, device=device)
        if z is None: z=torch.rand(4, device=device)

        # Keep params on the same device as the provided tensors (library sampling uses CUDA).
        self.xt = nn.Parameter(torch.as_tensor(xt, dtype=torch.float32, device=device).reshape(1))
        self.yt = nn.Parameter(torch.as_tensor(yt, dtype=torch.float32, device=device).reshape(1))
        self.a = nn.Parameter(torch.as_tensor(a, dtype=torch.float32, device=device).reshape(1))
        self.z = nn.Parameter(torch.as_tensor(z, dtype=torch.float32, device=device).reshape(4))

        # Filleted polyline: N vertices + N-2 fillet κ logits (κ = sigmoid(radii) ∈ (0,1)).
        # κ=0 sharp corner, κ=1 max feasible roundness for current edge lengths / angle.
        if vertices is None:
            # Short stroke along +x with small y jitter (meters, stroke-local)
            t = torch.linspace(0.0, 0.06, N_VERTICES, device=device)
            vertices = torch.stack([t, (torch.rand(N_VERTICES, device=device) - 0.5) * 0.02], dim=-1)
        if radii is None:
            # Random κ ~ U(0.2, 0.8) → logits
            k = torch.empty(N_VERTICES - 2, device=device).uniform_(0.2, 0.8)
            radii = _logit(k)

        self.vertices = nn.Parameter(vertices.float())
        self.radii = nn.Parameter(radii.float())  # logits for κ

        if not self.ink:
            self.color_transform = nn.Parameter(color)
        else:
            self.color_transform = torch.zeros(3).to(device)

    def forward(self, h, w, param2img, use_conv=True):
        # Do rigid body transformation
        full_param = self.get_path().unsqueeze(0) # 1 x 32 x 3

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

    def fillet_kappa(self):
        """Normalized fillet κ ∈ (0, 1) per interior corner."""
        return torch.sigmoid(self.radii)

    def fillet_radii_m(self):
        """Fillet radii in meters: κ * r_max(vertices)."""
        return radii_from_kappa(self.vertices, self.fillet_kappa())

    def make_valid(stroke):
        with torch.no_grad():
            stroke.vertices.data.clamp_(-0.15, 0.15)
            # κ = sigmoid(radii) is already in (0,1); keep logits numerically tame
            stroke.radii.data.clamp_(-8.0, 8.0)

            stroke.xt.data.clamp_(0,1.)
            stroke.yt.data.clamp_(0,1.)
            stroke.z.data.clamp_(0.02,0.98)
            if not stroke.ink:
                # Soft brushes splay on a hard touchdown; ink tips don't need this.
                stroke.z[0].data.clamp_(0.1,0.1)
            # stroke.z.data.clamp_(0.5,0.5)################3

            #stroke.color_transform.data.clamp_(0.02,0.75)
            if stroke.color_transform.min() < 0.35:
                # If it's a colored stroke, don't let it go to a flourescent color
                stroke.color_transform.data.clamp_(0.02,0.70)
            else:
                # Well balanced RGB, less constraint
                stroke.color_transform.data.clamp_(0.02,0.85)

    def clear_path_cache(self):
        self._path_cache = None

    def __getstate__(self):
        # Never pickle derived path cache (wrong device after load / .to()).
        state = super().__getstate__()
        state = dict(state)
        state['_path_cache'] = None
        return state

    def get_path(self):
        cache = getattr(self, '_path_cache', None)
        # Also drop cache if params moved devices without clearing.
        if cache is not None and cache.device == self.vertices.device:
            return cache
        self._path_cache = None

        if self.is_dot:
            path = torch.zeros((4,4), device=self.vertices.device, dtype=self.vertices.dtype)
            path[:,2] = 0.75
            self._path_cache = path
            return path

        # κ = sigmoid(radii) → meters via r_max(vertices). expand_stroke also clamps trims.
        #
        # Filleted expand_stroke is Python control flow with many host syncs; on CUDA
        # that dominates. Run the tiny (N≈4) geometry on CPU, then move the 32×2
        # samples back — same math, much less sync overhead.
        device = self.vertices.device
        if device.type == 'cpu':
            radii_m = radii_from_kappa(self.vertices, torch.sigmoid(self.radii))
            xy = sample_filleted_polyline(self.vertices, radii_m, n_points=N_PATH_POINTS)
        else:
            vertices_cpu = self.vertices.cpu()
            radii_m = radii_from_kappa(vertices_cpu, torch.sigmoid(self.radii.cpu()))
            xy = sample_filleted_polyline(
                vertices_cpu, radii_m, n_points=N_PATH_POINTS
            ).to(device)

        # Linearly interpolate the 4 z values into 32 for each waypoint on trajectory
        z = self.z[None, None]  # (1,1,4)
        z = torch.nn.functional.interpolate(z, size=N_PATH_POINTS, scale_factor=None, mode='linear')
        z = z[0, 0].clamp(self.MIN_STROKE_Z, 0.95)  # (32,)

        path = torch.cat([xy, z.unsqueeze(-1)], dim=-1)  # (32, 3)
        # Cache only for optimizable strokes (not robot/library execute with frozen params).
        if torch.is_grad_enabled() and (
            self.vertices.requires_grad or self.radii.requires_grad or self.z.requires_grad
        ):
            self._path_cache = path
        return path

    def execute(self, painter, x_start, y_start, rotation):
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


        stroke_complete = painter.move_to_trajectories(all_positions, all_orientations)

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

    def path_aabb(self):
        """Axis-aligned bounds of the dense XY path in stroke-local meters.

        Returns (x_min, y_min, x_max, y_max).
        """
        xy = self.get_path()[:, :2]
        mins = xy.min(dim=0).values
        maxs = xy.max(dim=0).values
        return mins[0].item(), mins[1].item(), maxs[0].item(), maxs[1].item()

    @classmethod
    def sample_for_library(cls, opt, family=None, device=None):
        """Sample a stroke from a structured shape family for library collection.

        Covers straight / arc / S / corner / hook / round-turn with stratified
        lengths and family-appropriate fillet κ. Lateral bend is length-relative.
        Pose (xt, yt, a) is left for the library packer to set.
        """
        # CPU is enough for library geometry; keeps execute/path packing simple.
        if device is None:
            device = 'cpu'
        if family is None:
            family = LIBRARY_SHAPE_FAMILIES[int(torch.randint(0, len(LIBRARY_SHAPE_FAMILIES), (1,)).item())]

        length = _sample_library_length(opt)
        vertices, kappa = _library_vertices_and_kappa(family, length, device)
        radii = _logit(kappa)
        try:
            min_z = float(opt.MIN_STROKE_Z)
        except Exception:
            min_z = 0.1
        z = _sample_library_z(device, min_z=min_z)

        try:
            ink = bool(opt.ink)
        except Exception:
            ink = False

        stroke = cls(
            opt,
            vertices=vertices,
            radii=radii,
            a=torch.zeros(1, device=device),
            xt=torch.zeros(1, device=device),
            yt=torch.zeros(1, device=device),
            z=z,
            ink=ink,
            device=device,
        )
        stroke.library_shape_family = family
        BrushStroke.make_valid(stroke)
        stroke.clear_path_cache()
        # Packing-friendly frame: x starts at 0, y centered on the execute origin.
        with torch.no_grad():
            x_min, y_min, x_max, y_max = stroke.path_aabb()
            stroke.vertices[:, 0] -= x_min
            stroke.vertices[:, 1] -= 0.5 * (y_min + y_max)
        stroke.clear_path_cache()
        return stroke
    
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

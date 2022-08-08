import torch
from torch import nn
import torchgeometry
import torchvision.transforms as T
import warnings
import numpy as np
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from continuous_brush_model import StrokeParametersToImage, special_sigmoid


strokes = np.load('/home/frida/ros_ws/src/intera_sdk/SawyerPainter/scripts/extended_stroke_library_intensities.npy') 

h, w = strokes.shape[1], strokes.shape[2]
h_og, w_og = h, w

# Crop
hs, he = int(.4*h), int(0.6*h)
ws, we = int(0.45*w), int(0.75*w)

pad_for_full = T.Pad((ws, hs, w_og-we, h_og-he))

param2img = StrokeParametersToImage(int(.6*h - .4*h), int(0.75*w - 0.45*w))
param2img.load_state_dict(torch.load(
    '/home/frida/ros_ws/src/intera_sdk/SawyerPainter/scripts/param2img.pt'))
param2img.eval()
param2img.to(device)

cos = torch.cos
sin = torch.sin
def rigid_body_transform(a, xt, yt, anchor_x, anchor_y):
    # a is the angle in radians, xt and yt are translation terms of pixels
    # anchor points are where to rotate around (usually the center of the image)
    # Blessed be Peter Schorn for the anchor point transform https://stackoverflow.com/a/71405577
    A = torch.zeros(1, 3, 3).to(device)
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
        weights = torch.zeros(3)
        weights[0] = a
        weights[1] = xt
        weights[2] = yt
        weights.requires_grad = True
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        anchor_x, anchor_y = w/2, h/2

        M = rigid_body_transform(self.weights[0], self.weights[1]*(w/2), self.weights[2]*(h/2), anchor_x, anchor_y)
        with warnings.catch_warnings(): # suppress annoing torchgeometry warning
            warnings.simplefilter("ignore")
            return torchgeometry.warp_perspective(x, M, dsize=(h,w))

class BrushStroke(nn.Module):
    def __init__(self, stroke_ind, 
                stroke_length=None, stroke_z=None, stroke_bend=None,
                color=None, 
                a=None, xt=None, yt=None):
        super(BrushStroke, self).__init__()

        if color is None: color=torch.rand(3)
        if a is None: a=(torch.rand(1)*2-1)*3.14
        if xt is None: xt=(torch.rand(1)*2-1)
        if yt is None: yt=(torch.rand(1)*2-1)


        if stroke_length is None: stroke_length=torch.rand(1)*(.05-.01) + .01
        if stroke_z is None: stroke_z = torch.rand(1)
        if stroke_bend is None: stroke_bend = torch.rand(1)*.04 - .02 
        stroke_bend = min(stroke_bend, stroke_length) if stroke_bend > 0 else max(stroke_bend, -1*stroke_length)

        self.transformation = RigidBodyTransformation(a, xt, yt)
        
        #self.stroke_ind = stroke_ind
        self.stroke_length = stroke_length
        self.stroke_z = stroke_z
        self.stroke_bend = stroke_bend 

        self.stroke_length.requires_grad = True
        self.stroke_z.requires_grad = True
        self.stroke_bend.requires_grad = True

        self.stroke_length = nn.Parameter(self.stroke_length)
        self.stroke_z = nn.Parameter(self.stroke_z)
        self.stroke_bend = nn.Parameter(self.stroke_bend)

        self.color_transform = nn.Parameter(color)

    def forward(self, strokes):
        # Do rigid body transformation
        full_param = torch.zeros((1,12)).to(device)
        
        # X
        full_param[0,0] = 0
        full_param[0,3] = self.stroke_length/3 
        full_param[0,6] = 2*self.stroke_length/3
        full_param[0,9] = self.stroke_length
        # Y
        full_param[0,1] = 0
        full_param[0,4] = self.stroke_bend
        full_param[0,7] = self.stroke_bend
        full_param[0,10] = 0
        # Z
        full_param[0,2] = 0.2
        full_param[0,5] = self.stroke_z
        full_param[0,8] = self.stroke_z
        full_param[0,11] = 0.2

        stroke = param2img(full_param).unsqueeze(0)
        stroke = pad_for_full(stroke)

        # Pad 1 or two to make it fit
        # print('ffff', strokes[0].shape)
        stroke = T.Resize((strokes[0].shape[0], strokes[0].shape[1]))(stroke)

        # x = self.transformation(strokes[self.stroke_ind].permute(2,0,1).unsqueeze(0))
        from plan_all_strokes import show_img
        # show_img(stroke)
        x = self.transformation(stroke)

        # Remove stray color from the neural network being sloppy
        # x[x < 0.1] = 0
        # # x[x >= 0.1] = 1
        #x = 1/(1+torch.exp(-1.*((x*2-1)+0.5) / 0.05))
        x = special_sigmoid(x)
        # import kornia as K
        # x = K.filters.median_blur(x, (3,3))
        

        # show_img(x)
        x = torch.cat([x,x,x,x], dim=1)
        # print('forawrd brush', x.shape)
        # Color change
        x = torch.cat((x[:,:3]*0 + self.color_transform[None,:,None,None], x[:,3:]), dim=1)
        return x

    def make_valid(stroke):
        with torch.no_grad():
            og_len = stroke.stroke_length.item()
            stroke.stroke_length.data.clamp_(0.01,0.05)
            # if stroke.stroke_length.item() != og_len:
            #     print('length constrained')
            
            stroke.stroke_bend.data.clamp_(-1*stroke.stroke_length, stroke.stroke_length)
            stroke.stroke_bend.data.clamp_(-.02,.02)

            stroke.stroke_z.data.clamp_(0.1,1.0)

            stroke.transformation.weights[1:3].data.clamp_(-1.,1.)

class Painting(nn.Module):
    def __init__(self, n_strokes, background_img=None, brush_strokes=None, unique_strokes=None):
        # h, w are canvas height and width in pixels
        super(Painting, self).__init__()
        self.n_strokes = n_strokes

        self.background_img = background_img

        if self.background_img.shape[1] == 3: # add alpha channel
            t =  torch.zeros((1,1,self.background_img.shape[2],self.background_img.shape[3])).to(device)
            # t[:,:3] = self.background_img
            self.background_img = torch.cat((self.background_img, t), dim=1)

        if brush_strokes is None:
            self.brush_strokes = nn.ModuleList([BrushStroke(np.random.randint(unique_strokes)) for _ in range(n_strokes)])
        else:
            self.brush_strokes = nn.ModuleList(brush_strokes)



    def forward(self, strokes, use_alpha=True):
        if self.background_img is None:
            canvas = torch.ones((1,4,strokes[0].shape[0],strokes[0].shape[1])).to(device)
        else:
            canvas = T.Resize(size=(strokes[0].shape[0],strokes[0].shape[1]))(self.background_img).detach()

        canvas[:,3] = 1 # alpha channel

        mostly_opaque = False#True

        for brush_stroke in self.brush_strokes:
            single_stroke = brush_stroke(strokes=strokes)

            if mostly_opaque: single_stroke[:,3][single_stroke[:,3] > 0.5] = 1.
            
            if use_alpha:
                canvas = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
            else:
                canvas = canvas[:,:3] * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke[:,:3]
        return canvas

    def to_csv(self):
        ''' To csv string '''
        csv = ''
        for bs in self.brush_strokes:
            # Translation in proportions from top left
            x = str((bs.transformation.weights[1].detach().cpu().item()+1)/2)
            y = str((bs.transformation.weights[2].detach().cpu().item()+1)/2)
            r = str(bs.transformation.weights[0].detach().cpu().item())
            # stroke_ind = str(bs.stroke_ind)
            length = str(bs.stroke_length.detach().cpu().item())
            thickness = str(bs.stroke_z.detach().cpu().item())
            bend = str(bs.stroke_bend.detach().cpu().item())
            color = bs.color_transform.detach().cpu().numpy()
            csv += ','.join([x,y,r,length,thickness,bend,str(color[0]),str(color[1]),str(color[2])])
            csv += '\n'
        csv = csv[:-1] # remove training newline
        return csv

    def validate(self):
        ''' Make sure all brush strokes have valid parameters '''
        for s in self.brush_strokes:
            BrushStroke.make_valid(s)
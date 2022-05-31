import torch
from torch import nn
import torchgeometry
import torchvision.transforms as T
import warnings
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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
    def __init__(self, stroke_ind, color=None, a=None, xt=None, yt=None):
        super(BrushStroke, self).__init__()

        if color is None: color=torch.rand(3)
        if a is None: a=(torch.rand(1)*2-1)*3.14
        if xt is None: xt=(torch.rand(1)*2-1)
        if yt is None: yt=(torch.rand(1)*2-1)

        self.transformation = RigidBodyTransformation(a, xt, yt)
        self.stroke_ind = stroke_ind
        self.color_transform = nn.Parameter(color)

    def forward(self, strokes):
        # Do rigid body transformation
        x = self.transformation(strokes[self.stroke_ind].permute(2,0,1).unsqueeze(0))
        # Color change
        x = torch.cat((x[:,:3]*0 + self.color_transform[None,:,None,None], x[:,3:]), dim=1)
        return x

class Painting(nn.Module):
    def __init__(self, n_strokes, background_img=None, brush_strokes=None):
        # h, w are canvas height and width in pixels
        super(Painting, self).__init__()
        self.n_strokes = n_strokes

        self.background_img = background_img

        if self.background_img.shape[1] == 3: # add alpha channel
            t =  torch.zeros((1,1,self.background_img.shape[2],self.background_img.shape[3])).to(device)
            # t[:,:3] = self.background_img
            self.background_img = torch.cat((self.background_img, t), dim=1)

        if brush_strokes is None:
            self.brush_strokes = nn.ModuleList([BrushStroke(np.random.randint(len(strokes_full))) for _ in range(n_strokes)])
        else:
            self.brush_strokes = nn.ModuleList(brush_strokes)



    def forward(self, strokes):
        if self.background_img is None:
            canvas = torch.ones((1,4,strokes[0].shape[0],strokes[0].shape[1])).to(device)
        else:
            canvas = T.Resize(size=(strokes[0].shape[0],strokes[0].shape[1]))(self.background_img).detach()

        canvas[:,3] = 1 # alpha channel

        for brush_stroke in self.brush_strokes:
            single_stroke = brush_stroke(strokes=strokes)
            canvas = canvas * (1 - single_stroke[:,3:]) + single_stroke[:,3:] * single_stroke
        return canvas

    def to_csv(self):
        ''' To csv string '''
        csv = ''
        for bs in self.brush_strokes:
            # Translation in proportions from top left
            x = str((bs.transformation.weights[1].detach().cpu().item()+1)/2)
            y = str((bs.transformation.weights[2].detach().cpu().item()+1)/2)
            r = str(bs.transformation.weights[0].detach().cpu().item())
            stroke_ind = str(bs.stroke_ind)
            color = bs.color_transform.detach().cpu().numpy()
            csv += ','.join([x,y,r,stroke_ind,str(color[0]),str(color[1]),str(color[2])])
            csv += '\n'
        csv = csv[:-1] # remove training newline
        return csv

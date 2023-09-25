
##########################################################
#################### Copyright 2022 ######################
################ by Vihaan Misra ##################
### The Robotics Institute, Carnegie Mellon University ###
################ All rights reserved. ####################
##########################################################

'''
This is modified code from: https://github.com/mtli/PhotoSketch
'''


import os
import torch
import torch.nn as nn
import torch.nn.parallel
from .pix2pix import ResnetGenerator
from PIL import Image
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_requires_grad(params, flag):
    for p in params:
        p.requires_grad = flag

class RepeatChannel(nn.Module):
    def __init__(self, repeat):
        super(RepeatChannel, self).__init__()
        self.repeat = repeat

    def forward(self, img):
        return img.repeat(1, self.repeat, 1, 1)


class Downsample(nn.Module):
    def __init__(self, n_iter):
        super(Downsample, self).__init__()
        self.n_iter = n_iter

    def forward(self, img):
        for _ in range(self.n_iter):
            img = nn.functional.interpolate(img, scale_factor=0.5, mode='bicubic')
        return img


class Upsample(nn.Module):
    def __init__(self, n_iter):
        super(Upsample, self).__init__()
        self.n_iter = n_iter

    def forward(self, img):
        for _ in range(self.n_iter):
            img = nn.functional.interpolate(img, scale_factor=2.0, mode='bicubic')
        return img

class OutputTransform(nn.Module):
    def __init__(self, path, process='', diffaug_policy=''):
        super(OutputTransform, self).__init__()
        self.photosketch_path = path
        self.augment = None

        transforms = []
        process = process.split(',')
        for p in process:
            if p.startswith('up'):
                n_iter = int(p.replace('up', ''))
                transforms.append(Upsample(n_iter))
            elif p.startswith('down'):
                n_iter = int(p.replace('down', ''))
                transforms.append(Downsample(n_iter))
            elif p == 'to3ch':
                transforms.append(RepeatChannel(3))
            elif p == 'toSketch':
                sketch = self.setup_sketch(self.photosketch_path)
                transforms.append(sketch)
            else:
                ValueError("Transforms contains unrecognizable key: %s" % p)
        self.transforms = nn.Sequential(*transforms)

    def setup_sketch(self, photosketch_path):
        sketch = ResnetGenerator(3, 1, n_blocks=9, use_dropout=False)

        state_dict = torch.load(photosketch_path, map_location='cpu')
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata

        sketch.load_state_dict(state_dict)
        sketch.train()
        set_requires_grad(sketch.parameters(), False)
        return sketch

    def forward(self, img, apply_aug=True):
        img = self.transforms(img)
        return img



#import pathlib
#working_dir = pathlib.Path().resolve()
#path = '/home/frida/ros_ws/src/intera_sdk/SawyerPainter/src/sketch_loss/pretrained/photosketch.pth'#os.path.join(working_dir, 'pretrained/photosketch.pth')

tf_real = None

def compute_sketch_loss(sketch, painting, comparator=torch.nn.MSELoss(), writer=None, it=0):
    # path = '/mnt/Data1/vmisra/Frida/scripts/pretrained/photosketch.pth'
    # img_path = '/mnt/Data1/vmisra/Frida/scripts/frida.jpg'
    # img = Image.open(img_path)
    # tensor_transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    # width, height = img.size
    # img_tensor = tensor_transform(img).float()
    # img_tensor = img_tensor.reshape(1, 3, height, width)
    from plan import format_img
    # print(painting.min().item(), painting.max().item(), sketch.min().item(), sketch.max().item())

    if tf_real is None:
        
        try: # If running ros, get the painting code dir
            import rospkg
            rospack = rospkg.RosPack()
            # get the file path for painter code
            ros_dir = rospack.get_path('paint')
            path = os.path.join(ros_dir,'src','sketch_loss', 'pretrained', 'photosketch.pth')
        except:
            path = 'sketch_loss/pretrained/photosketch.pth'
        t_real = 'toSketch'
        tf_real = OutputTransform(path, process=t_real).to(device)
    sketch_tensor = (tf_real(sketch*2-1)+1)/2

    # image_transform = transforms.Compose([
    # transforms.ToPILImage()])

    # sketch_style = image_transform(sketch_tensor[0])

    sketch_p_tensor = (tf_real(painting*2-1)+1)/2
    # print(sketch_p_tensor.min().item(), sketch_p_tensor.max().item(), sketch_tensor.min().item(), sketch_tensor.max().item())
    # print(sketch_p_tensor.mean().item(), sketch_p_tensor.std().item(), sketch_tensor.mean().item(), sketch_tensor.std().item())
    # print(format_img((sketch_p_tensor)).mean(), format_img((sketch_p_tensor)).max())
    if writer is not None and it%5 == 0:
        writer.add_image('images/painting_sketch', format_img((sketch_p_tensor))*255., it)
    if writer is not None and it == 1: writer.add_image('images/sketch_sketch', format_img((sketch_tensor))*255., it)

    # sketch_painting = image_transform(sketch_p_tensor[0])

    # return ((sketch_tensor - sketch_p_tensor)**2).mean()
    return comparator(sketch_tensor, sketch_p_tensor)

def compute_canny_loss(sketch, painting, comparator=torch.nn.MSELoss(), writer=None, it=0):
    from kornia.filters import canny
    from plan import format_img
    canny_p = canny(painting)[0]
    canny_sketch = canny(sketch)[0]

    if writer is not None and it%5 == 0:
        writer.add_image('images/painting_sketch', format_img((canny_p))*255., it)
    if writer is not None and it == 1: writer.add_image('images/sketch_sketch', format_img((canny_sketch))*255., it)

    return comparator(canny_p, canny_sketch)
    # return -1. * (canny_p*canny_sketch).mean() # opposite of union
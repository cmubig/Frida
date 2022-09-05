import torch
import torch.nn as nn
import torch.nn.parallel
from pix2pix import ResnetGenerator
from PIL import Image
import torchvision.transforms as transforms




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

def compute_sketch_loss(style, painting):
    path = '/mnt/Data1/vmisra/Frida/scripts/pretrained/photosketch.pth'
    t_real = 'toSketch'
    tf_real = OutputTransform(path, process=t_real)
    # img_path = '/mnt/Data1/vmisra/Frida/scripts/frida.jpg'
    # img = Image.open(img_path)
    # tensor_transform = transforms.Compose([
    #     transforms.PILToTensor()
    # ])
    # width, height = img.size
    # img_tensor = tensor_transform(img).float()
    # img_tensor = img_tensor.reshape(1, 3, height, width)
    sketch_tensor = tf_real(style)

    # image_transform = transforms.Compose([
    # transforms.ToPILImage()])

    # sketch_style = image_transform(sketch_tensor[0])

    sketch_p_tensor = tf_real(painting)

    # sketch_painting = image_transform(sketch_p_tensor[0])

    return ((sketch_tensor - sketch_p_tensor)**2).mean()
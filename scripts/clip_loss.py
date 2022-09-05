'''
Adapted from CLIPasso
https://github.com/yael-vinker/CLIPasso/blob/173b954de3298f43a860bf38ea6387916e67ff6c/models/loss.py
@misc{vinker2022clipasso,
      title={CLIPasso: Semantically-Aware Object Sketching},
      author={Yael Vinker and Ehsan Pajouheshgar and Jessica Y. Bo and Roman Christian Bachmann and Amit Haim Bermano and Daniel Cohen-Or and Amir Zamir and Ariel Shamir},
      year={2022},
      eprint={2202.05822},
      archivePrefix={arXiv},
      primaryClass={cs.GR}
}
'''

import collections
import torch
import torch.nn as nn
from torchvision import models, transforms
import clip
import warnings



class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = None

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x):
        self.featuremaps = collections.OrderedDict()
        fc_features = self.clip_model.encode_image(x).float()
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps


def l2_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def l1_layers(xs_conv_features, ys_conv_features, clip_model_name):
    return [torch.abs(x_conv - y_conv).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


def cos_layers(xs_conv_features, ys_conv_features, clip_model_name):
    if "RN" in clip_model_name:
        return [torch.square(x_conv, y_conv, dim=1).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    return [(1 - torch.cosine_similarity(x_conv, y_conv, dim=1)).mean() for x_conv, y_conv in
            zip(xs_conv_features, ys_conv_features)]


class CLIPConvLoss(torch.nn.Module):
    def __init__(self, args):
        super(CLIPConvLoss, self).__init__()
        self.clip_model_name = args.clip_model_name
        assert self.clip_model_name in [
            "RN50",
            "RN101",
            "RN50x4",
            "RN50x16",
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.clip_conv_loss_type = args.clip_conv_loss_type
        self.clip_fc_loss_type = "Cos"  # args.clip_fc_loss_type
        assert self.clip_conv_loss_type in [
            "L2", "Cos", "L1",
        ]
        assert self.clip_fc_loss_type in [
            "L2", "Cos", "L1",
        ]

        self.distance_metrics = \
            {
                "L2": l2_layers,
                "L1": l1_layers,
                "Cos": cos_layers
            }

        self.model, clip_preprocess = clip.load(
            self.clip_model_name, args.device, jit=False)

        if self.clip_model_name.startswith("ViT"):
            self.visual_encoder = CLIPVisualEncoder(self.model)

        else:
            self.visual_model = self.model.visual
            layers = list(self.model.visual.children())
            init_layers = torch.nn.Sequential(*layers)[:8]
            self.layer1 = layers[8]
            self.layer2 = layers[9]
            self.layer3 = layers[10]
            self.layer4 = layers[11]
            self.att_pool2d = layers[12]

        self.args = args

        self.img_size = clip_preprocess.transforms[1].size
        self.model.eval()
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            clip_preprocess.transforms[0],  # Resize
            clip_preprocess.transforms[1],  # CenterCrop
            clip_preprocess.transforms[-1],  # Normalize
        ])

        self.model.eval()
        self.device = args.device
        self.num_augs = self.args.num_aug_clip

        augemntations = []
        if "affine" in args.augemntations:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)

        self.clip_fc_layer_dims = None  # self.args.clip_fc_layer_dims
        self.clip_conv_layer_dims = None  # self.args.clip_conv_layer_dims
        self.clip_fc_loss_weight = args.clip_fc_loss_weight
        self.counter = 0

    def forward(self, sketch, target, mode="train"):
        """
        Parameters
        ----------
        sketch: Torch Tensor [1, C, H, W]
        target: Torch Tensor [1, C, H, W]
        """
        #         y = self.target_transform(target).to(self.args.device)
        conv_loss_dict = {}
        x = sketch.to(self.device)
        y = target.to(self.device)
        sketch_augs, img_augs = [self.normalize_transform(x)], [
            self.normalize_transform(y)]
        if mode == "train":
            for n in range(self.num_augs):
                with warnings.catch_warnings():
                    # RandomPerspective has a really annoying warning
                    warnings.simplefilter("ignore")
                    augmented_pair = self.augment_trans(torch.cat([x, y]))
                sketch_augs.append(augmented_pair[0].unsqueeze(0))
                img_augs.append(augmented_pair[1].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0).to(self.device)
        ys = torch.cat(img_augs, dim=0).to(self.device)

        if self.clip_model_name.startswith("RN"):
            xs_fc_features, xs_conv_features = self.forward_inspection_clip_resnet(
                xs.contiguous())
            ys_fc_features, ys_conv_features = self.forward_inspection_clip_resnet(
                ys.detach())

        else:
            xs_fc_features, xs_conv_features = self.visual_encoder(xs)
            ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        conv_loss = self.distance_metrics[self.clip_conv_loss_type](
            xs_conv_features, ys_conv_features, self.clip_model_name)

        for layer, w in enumerate(self.args.clip_conv_layer_weights):
            if w:
                conv_loss_dict[f"clip_conv_loss_layer{layer}"] = conv_loss[layer] * w

        if self.clip_fc_loss_weight:
            # fc distance is always cos
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features,
                       ys_fc_features, dim=1)).mean()
            conv_loss_dict["fc"] = fc_loss * self.clip_fc_loss_weight

        self.counter += 1
        return conv_loss_dict

    def forward_inspection_clip_resnet(self, x):
        def stem(m, x):
            for conv, bn in [(m.conv1, m.bn1), (m.conv2, m.bn2), (m.conv3, m.bn3)]:
                x = m.relu(bn(conv(x)))
            x = m.avgpool(x)
            return x
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = stem(self.visual_model, x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]


def get_image_augmentation(use_normalized_clip):
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    return augment_trans

augment_trans = get_image_augmentation(False)
num_augs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def clip_loss(img0, img1):
    img0_batch = torch.cat([augment_trans(img0) for n in range(num_augs)])
    img1_batch = torch.cat([augment_trans(img1) for n in range(num_augs)])
    img0_features = clip_model.encode_image(img0_batch)
    img1_features = clip_model.encode_image(img1_batch)

    loss = 0
    for n in range(num_augs):
        loss -= torch.cosine_similarity(img0_features[n:n+1], img1_features[n:n+1], dim=1)[0] / num_augs
    return loss
class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

# clip_conv_layer_weights = [1.0, 1.0, 1.0, 1.0, 0]
clip_conv_layer_weights = [0, 0, 0, 0, 1.0]
a = {'clip_model_name':'ViT-B/32','clip_conv_loss_type':'Cos','device':device,
    'num_aug_clip':num_augs,'augemntations':['affine'],
     'clip_fc_loss_weight':0.0,'clip_conv_layer_weights':clip_conv_layer_weights}
clip_conv_loss_model = CLIPConvLoss(Dict2Class(a))

def clip_conv_loss(painting, target):
    loss = 0
    clip_loss = clip_conv_loss_model(painting[:,:3], target)
    for key in clip_loss.keys():
        loss += clip_loss[key]
    return loss



import torchvision.transforms as transforms

augment_trans_text = transforms.Compose([
    # transforms.GaussianBlur((21,21), sigma=(1.)),
    # transforms.GaussianBlur((15,15), sigma=(1.)), # Doesn't work well it appears
    # transforms.GaussianBlur((5,5), sigma=(1.)), # Maybe better than nothing? but marginally
    # transforms.GaussianBlur((15,15), sigma=(2.)),
    # transforms.Resize(64),
    # transforms.Resize(256),
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

import clip
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
clip_model_16, preprocess_16 = clip.load('ViT-B/16', device, jit=False)

def clip_text_loss(p, text_features, num_augs, text_features_16=None):
    loss = 0
    img_augs = []
    with warnings.catch_warnings():
        # RandomPerspective has a really annoying warning
        warnings.simplefilter("ignore")
        for n in range(num_augs):
            img_augs.append(augment_trans_text(p[:,:3]))

    im_batch = torch.cat(img_augs)
    # from plan_all_strokes import show_img 
    # show_img(im_batch[0])
    image_features = clip_model.encode_image(im_batch)
    if text_features_16 is not None:
        image_features_16 = clip_model.encode_image(im_batch)
    for n in range(num_augs):
        loss -= torch.cosine_similarity(text_features, image_features[n:n+1], dim=1)
        if text_features_16 is not None:
            loss -= torch.cosine_similarity(text_features_16, image_features_16[n:n+1], dim=1)

    return loss / num_augs if text_features_16 is not None else loss / (2*num_augs)

def clip_fc_loss(p, other_img, num_augs, text_features_16=None):
    loss = 0
    p_augs, other_img_augs = [], []
    with warnings.catch_warnings():
        # RandomPerspective has a really annoying warning
        warnings.simplefilter("ignore")
        for n in range(num_augs):
            p_augs.append(augment_trans_text(p[:,:3]))
            other_img_augs.append(augment_trans_text(other_img[:,:3]))

    p_batch = torch.cat(p_augs)
    other_batch = torch.cat(other_img_augs)
    p_features = clip_model.encode_image(p_batch)
    other_features = clip_model.encode_image(other_batch)

    for n in range(num_augs):
        loss -= torch.cosine_similarity(other_features[n:n+1], p_features[n:n+1], dim=1)

    return loss / num_augs if text_features_16 is not None else loss / (2*num_augs)
'''
An adaption of https://github.com/robinszym/EmotionPredictor
'''

import os
import clip
import torch
from torch import nn
import torchvision.transforms as transforms
import warnings

device = 'cuda' if torch.cuda.is_available() else 'cpu'

augment = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(224, scale=(0.7,0.9), antialias=True),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

ARTEMIS_EMOTIONS = [
    'amusement',
    'awe',
    'contentment',
    'excitement',
    'anger',
    'disgust',
    'fear',
    'sadness',
    'something else'
]

class SLP(nn.Module):
    def __init__(self,input_size = 512, output_size = 9):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x):
        fp = self.layers(x.float())
        return fp

root = os.path.dirname(os.path.realpath(__file__))

MODEL_PATH = os.path.join(root, "pretrained_emotion_models/C-ViT-B32")
# MODEL_PATH = "pretrained_emotion_models/C-RN101"
# MODEL_PATH = "pretrained_emotion_models/C-RN50"
# MODEL_PATH = "pretrained_emotion_models/C-RN50x16"
# MODEL_PATH = "pretrained_emotion_models/C-RN50x4"
# emotion = SLP(768)
emotion = SLP(512) # For ViT
emotion.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
emotion = emotion.to(device)

# clip_model, preprocess = clip.load("RN50x16")
clip_model, preprocess = clip.load("ViT-B/32")
clip_model.eval()
clip_model = clip_model.to(device)



def emotion_loss(painting, emotion_feats, num_augs):
    loss = 0
    img_augs = []
    with warnings.catch_warnings():
        # RandomPerspective has a really annoying warning
        warnings.simplefilter("ignore")
        for n in range(num_augs):
            img_augs.append(augment(painting[:,:3]))

    im_batch = torch.cat(img_augs)
    image_features = emotion(clip_model.encode_image(im_batch))
    for n in range(num_augs):
        loss -= torch.cosine_similarity(emotion_feats, image_features[n:n+1], dim=1)

    return loss / num_augs


if __name__ == '__main__':
    img = torch.rand(1, 3, 256, 256).to(device)
    target_emotions = torch.tensor([0,0,0,0,0,0,0,1,0]).float().to(device)
    emotion_loss(img, target_emotions, num_augs=4)


    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    def show_img(img, display_actual_size=True):
        if type(img) is torch.Tensor:
            img = img.detach().cpu().numpy()

        img = img.squeeze()
        if img.shape[0] < 5:
            img = img.transpose(1,2,0)

        if img.max() > 4:
            img = img / 255.
        img = np.clip(img, a_min=0, a_max=1)

        if display_actual_size:
            # Display at actual size: https://stackoverflow.com/questions/60144693/show-image-in-its-original-resolution-in-jupyter-notebook
            # Acquire default dots per inch value of matplotlib
            dpi = matplotlib.rcParams['figure.dpi']
            # Determine the figures size in inches to fit your image
            height, width = img.shape[0], img.shape[1]
            figsize = width / float(dpi), height / float(dpi)

            plt.figure(figsize=figsize)

        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.xticks([]), plt.yticks([])
        #plt.scatter(img.shape[1]/2, img.shape[0]/2)
        plt.show()

    img.requires_grad = True
    optim = torch.optim.Adam([img])
    for it in range(200):
        optim.zero_grad()
        loss = emotion_loss(img, target_emotions, num_augs=10)
        loss.backward()
        optim.step()
    show_img(img[0])

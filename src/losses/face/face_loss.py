# from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import warnings

augment = transforms.Compose([
    # transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    # transforms.RandomResizedCrop(224, scale=(0.7,0.9)),
    # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# # If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)

# Create an inception resnet (in eval mode):
# facenet = InceptionResnetV1(pretrained='vggface2').eval()
# facenet = nn.Sequential(*list(facenet.children())[:2]).to('cuda')

# print(facenet)
def parse_face_data(img):
    # norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    with torch.no_grad():
        print('face feat shape', facenet((img)).shape)
        return facenet(img)

def face_loss(painting, input_face_feats, num_augs):
    num_augs = 1
    loss = 0
    img_augs = []
    with warnings.catch_warnings():
        # RandomPerspective has a really annoying warning
        warnings.simplefilter("ignore")
        for n in range(num_augs):
            img_augs.append(augment(painting[:,:3]))

    im_batch = torch.cat(img_augs)
    image_features = facenet(im_batch)
    # for n in range(num_augs):
    #     # loss -= torch.cosine_similarity(input_face_feats, image_features[n:n+1], dim=1)
    #     loss -= torch.cosine_similarity(torch.flatten(input_face_feats, start_dim=1), torch.flatten(image_features[n:n+1], start_dim=1), dim=1)
    loss = nn.MSELoss()(torch.flatten(input_face_feats, start_dim=1), torch.flatten(image_features, start_dim=1))
    # print(loss)
    return loss / num_augs
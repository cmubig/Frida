from diffusers import StableDiffusionPipeline
import torch
import torchvision.transforms as transforms

# Thank you Ajay Jain for vectorFusion which forms this theoretical basis
# and Thank you to the author of this notebook for implmentation details https://colab.research.google.com/drive/1pT-WPdcdQ7e4E2EkkwGiPTIL1ZNMyj11#scrollTo=fXn5P2R60EuQ

try:
    stable_pipe
except NameError:
    stable_pipe = None

device = 'cuda' if torch.cuda.is_available() else 'cpu'


augment_trans = transforms.Compose([
    transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
    transforms.RandomResizedCrop(512, scale=(0.7,0.9)),
])

def load_stable_pipe():
    stable_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                    revision="fp16", torch_dtype=torch.float16,
                                                    use_auth_token=True)
    stable_pipe = stable_pipe.to(device)
    stable_pipe.scheduler.set_timesteps(1000)
    # del stable_pipe.safety_checker
    # stable_pipe.safety_checker = lambda clip_input, images: (images, [False for _ in images])  # haha
    return stable_pipe

def encode_text_stable_diffusion(text):
    global stable_pipe
    if stable_pipe is None:
        stable_pipe = load_stable_pipe()
    toks = stable_pipe.tokenizer([" ", text], padding=True)
    text_embeddings = stable_pipe.text_encoder(
        torch.LongTensor(toks.input_ids).to(device),
        attention_mask=torch.tensor(toks.attention_mask).to(device))[0]
    return text_embeddings

def stable_diffusion_loss(img_cut, text_embeddings, guidance_scale=70.0, num_augs=1):
    global stable_pipe
    if stable_pipe is None:
        stable_pipe = load_stable_pipe()

    # Augment
    img_augs = []
    for n in range(num_augs):
        img_augs.append(augment_trans(img_cut))
    img_cut = torch.cat(img_augs) * 2 - 1

    img_cut = torch.nn.functional.interpolate(img_cut, (512, 512))#############################
    # img_cut = torch.nn.functional.interpolate(img_cut, (400, 400))

    with torch.autocast(device):
        encoded = stable_pipe.vae.encode(img_cut.half()).latent_dist.sample() * 0.18215
    with torch.inference_mode(), torch.autocast(device):
        noise = torch.randn_like(encoded)
        ts = stable_pipe.scheduler.timesteps
        t = torch.randint(2, len(ts) - 1, (len(img_cut),)).long()
        t = ts[t]
        noised = stable_pipe.scheduler.add_noise(encoded, noise, t)
        pred_noise = stable_pipe.unet(noised.repeat(2, 1, 1, 1).half().to(device),
                                      t.repeat(2).to(device),
                                      text_embeddings.repeat_interleave(noised.shape[0], dim=0)).sample
        noise_pred_uncond, noise_pred_text = pred_noise.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    loss = (encoded.flatten() * (noise_pred - noise).flatten().clone()).mean()
    alpha_bar = stable_pipe.scheduler.alphas_cumprod[t]
    beta = torch.sqrt(1 - alpha_bar)
    weights = beta.pow(2)
    weigths = weights * 0 + 1  # TODO
    # return (loss * weights.to(device)).mean()
    l = (loss * weights.to(device)).mean()
    # print(l.item())
    return l
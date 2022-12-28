import torch

from .sd import StableDiffusion

class StableDiffusionLoss():
    def __init__(self):
        self.fp16 = True ################
        # self.guidance = StableDiffusion('cuda', '2.0')
        self.guidance = StableDiffusion('cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        for p in self.guidance.parameters():
            p.requires_grad = False

    def encode_text_stable_diffusion(self, text):
        text_z = self.guidance.get_text_embeds([text], [''])
        return text_z

    def stable_diffusion_loss(self, pred_rgb, text_z):
        with torch.cuda.amp.autocast(enabled=self.fp16):
            loss = self.guidance.train_step(text_z, pred_rgb)
        # return loss
        # print(loss)
        return loss#self.scaler.scale(loss)

    def stable_diffusion_step(self, loss, optim):
        self.scaler.scale(loss).backward()
        self.scaler.step(optim)
        self.scaler.update()

# guidance = StableDiffusion('cuda', '2.0')
# scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

# for p in guidance.parameters():
#     p.requires_grad = False


# def encode_text_stable_diffusion(text):
#     text_z = self.guidance.get_text_embeds([text], [self.opt.negative])
#     return text_z

# def stable_diffusion_loss(pred_rgb, text_z):
#     with torch.cuda.amp.autocast(enabled=self.fp16):
#         loss = guidance.train_step(text_z, pred_rgb)
#     # return loss
#     return scaler.scale(loss)

# def stable_diffusion_step(loss, optim):
#     scaler.scale(loss).backward()
#     scaler.step(optim)
#     scaler.update()
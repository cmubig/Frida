import torch
import tensordict

class CanvasDataset(torch.utils.data.Dataset):
    def __init__(self, opt, datapath):
        self.opt = opt
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

        data = torch.load(datapath)
        self.current_canvas = data['current_canvas']
        self.target_img = data['target_img']
        self.remaining_strokes = data['remaining_strokes']
        self.strokes_tensor = data['strokes']

        self.indices = torch.arange(self.current_canvas.shape[0])
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        current_canvas = self.current_canvas[idx]
        target_img = self.target_img[idx]
        remaining_strokes = self.remaining_strokes[idx]
        strokes_tensor = self.strokes_tensor[idx]

        return current_canvas, target_img, remaining_strokes, strokes_tensor

class SequentialCanvasDataset(torch.utils.data.Dataset):
    def __init__(self, opt, datapath):
        self.opt = opt
        self.h_render = int(opt.render_height)
        self.w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

        data = torch.load(datapath)
        self.canvases = data['canvases']
        self.strokes_tensor = data['strokes']

        self.indices = torch.arange(self.canvases.shape[0])
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Shape batch_size, n_strokes+1, 3, h_render, w_render
        current_canvas = self.canvases[idx]
        
        # Shape batch_size, n_strokes, 9
        strokes_tensor = self.strokes_tensor[idx]

        return current_canvas, strokes_tensor
    
    def load_data(self, datapath):
        data = torch.load(datapath)

        incoming_canvases = data['canvases']
        incoming_strokes = data['strokes']
        current_max_strokes = self.canvases.shape[1]
        incoming_max_strokes = incoming_canvases.shape[1]

        # Pad to greater of max
        if incoming_max_strokes > current_max_strokes:
            canv_padding = -1*torch.ones(self.canvases.shape[0],incoming_max_strokes-current_max_strokes,3,self.h_render,self.w_render)
            self.canvases = torch.cat([canv_padding, self.canvases],dim=1)
            self.canvases = torch.cat([self.canvases, incoming_canvases],dim=0)

            stroke_padding = -1*torch.ones(self.strokes_tensor.shape[0],incoming_max_strokes-current_max_strokes,9)
            self.strokes_tensor = torch.cat([stroke_padding, self.strokes_tensor],dim=1)
            self.strokes_tensor = torch.cat([self.strokes_tensor, incoming_strokes],dim=0)
        else:
            canv_padding = -1*torch.ones(incoming_canvases.shape[0],current_max_strokes-incoming_max_strokes,3,self.h_render,self.w_render)
            incoming_canvases = torch.cat([canv_padding, incoming_canvases],dim=1)
            self.canvases = torch.cat([self.canvases, incoming_canvases],dim=0)

            stroke_padding = -1*torch.ones(incoming_strokes.shape[0],current_max_strokes-incoming_max_strokes,9)
            incoming_strokes = torch.cat([stroke_padding, incoming_strokes],dim=1)
            self.strokes_tensor = torch.cat([self.strokes_tensor, incoming_strokes],dim=0)

        self.indices = torch.arange(self.canvases.shape[0])
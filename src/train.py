import os
import sys
import datetime
import argparse
import random

import torch
from matplotlib import pyplot as plt

from options import Options
from painting import Painting
from brush_stroke import BrushStroke
from my_tensorboard import TensorBoard
from stroke_generator import StrokeGenerator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_canvas_as_img(canvas, path='outputs', suffix=''):
    plt.imshow(canvas.detach().cpu().permute(1,2,0).numpy())
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            raise Exception(f"Path provided for saving canvas does not exist and couldn't be created.\n\t{path}")
    plt.savefig(os.path.join(path,f"canvas{suffix}.png"))

if __name__ == '__main__':
    opt = Options()
    # Initialize opt to set some in code
    opt.parser = opt.initialize(argparse.ArgumentParser(description="FRIDA Robot Painter"))
    opt.CANVAS_WIDTH_M = 0.18
    opt.CANVAS_HEIGHT_M = opt.CANVAS_WIDTH_M * 0.667
    opt.gather_options()

    # Setup logging
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    # Model and optimizer
    model = StrokeGenerator(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch_size = 64
    epochs = 100
    
    # Set up canvas
    n_strokes = 1
    w_render = int(opt.render_height * (opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))
    h_render = int(opt.render_height)
    opt.w_render, opt.h_render = w_render, h_render
    blank_canvas = torch.ones(batch_size,3,h_render,w_render).to(device)
    # Shape [batch, channel[r,g,b], h, w]
    # Note: Alphas are not being used

    n_strokes = 1 # Strokes per canvas

    for e in range(epochs):
        strokes_gt = torch.zeros((batch_size,n_strokes,9))
        canvas_gt = torch.zeros_like(blank_canvas)

        for b in range(batch_size):
            brush_strokes = []

            xys = [(x,y) for x in (torch.rand(n_strokes)-0.5)*1.9 \
                        for y in (torch.rand(n_strokes)-0.5)*1.9]
            for i in range(len(xys)):
                x,y = xys[i]
                brush_stroke = BrushStroke(opt, xt=x, yt=y, ink=None)
                brush_strokes.append(brush_stroke)

                # Model outputs [l, z, b, x, y, a, r, g, b] 
                strokes_gt[b,i,0] = brush_stroke.stroke_length
                strokes_gt[b,i,1] = brush_stroke.stroke_z
                strokes_gt[b,i,2] = brush_stroke.stroke_bend
                strokes_gt[b,i,3:6] = torch.tensor([x,y,brush_stroke.transformation.a])
                strokes_gt[b,i,6:] = brush_stroke.color_transform

            # Paint ground truth brush stroke and save image
            painting = Painting(
                opt, 
                n_strokes=0, # Inits with 0 random brush strokes
                background_img=blank_canvas[b].unsqueeze(0),
                brush_strokes=brush_strokes # Adds strokes passed in instead
            ).to(device)
            canvas_gt[b] = painting(opt.h_render, opt.w_render, use_alpha=False, return_alphas=False)[0]

        # Forward pass
        stroke_pred = model(
            current_canvas=blank_canvas,
            target_img=canvas_gt,
            target_txt="A single stroke",
            remaining_strokes=1,
            color_palette=None # Not implemented
        )

        loss = torch.nn.functional.mse_loss(stroke_pred, strokes_gt)
        loss.backward()
        optim.step()

        print(f"Epoch: {e} Loss: {loss.item():0.2f}")
        print(loss)

        # Paint ground truth brush stroke and save image
        if e%10==0:
            # Model outputs [l, z, b, x, y, a, r, g, b] 
            generated_stroke = BrushStroke(opt, 
                                           stroke_length=stroke_pred[0,0],
                                           stroke_z=stroke_pred[0,1],
                                           stroke_bend=stroke_pred[0,2],
                                           color=stroke_pred[0,6:],
                                           a=stroke_pred[0,5],
                                           xt=stroke_pred[0,3], 
                                           yt=stroke_pred[0,4], 
                                           ink=None)
            generated_painting = Painting(
                opt, n_strokes=0, background_img=blank_canvas[0,], brush_strokes=[generated_stroke]
            ).to(device)
            generated_canvas, _ = generated_painting(opt.h_render, opt.w_render, use_alpha=False, return_alphas=True)
            
            save_canvas_as_img(canvas_gt[0], suffix=f'_{run_name}_target_{e}')
            save_canvas_as_img(generated_canvas[0], suffix=f'_{run_name}_generated_{e}')
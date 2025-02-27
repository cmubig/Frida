import os
os.environ['QT_QPA_PLATFORM']='offscreen' # Helps with matplotlib on some systems. Disables plotting on screen 

import sys
import torch 
import argparse
import datetime
from matplotlib import pyplot as plt

from options import Options
from my_tensorboard import TensorBoard
from brush_stroke import BrushStroke, BrushStrokeBatch
from painting import Painting, PaintingBatch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

###################
# Hyperparameters #
###################
batch_size = 16 # Make divisible by 4
n_strokes = 3

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

    # Setup dims
    h_render = int(opt.render_height)
    w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

    # Setup canvases
    blank_canvas = torch.ones(batch_size,3,h_render,w_render).to(device)

    # Generate a batch of random brush strokes
    brush_stroke_batches = []
    for i in range(n_strokes):
        brush_stroke_batches.append(BrushStrokeBatch(opt, batch_size, ink=None, init_differentiably=False).to(device))

    # Initialize painting
    painting = PaintingBatch(opt, batch_size, background_img=blank_canvas).to(device)
    
    # Paint brush stroke[s]
    canvas_gt = painting(brush_stroke_batches, h_render, w_render, use_alpha=False, return_alphas=False)

    # Plot and save canvases
    fig, axs = plt.subplots(batch_size//4,4)
    i = 0
    I,J = axs.shape
    for i in range(I):
        for j in range(J):
            ax = axs[i,j]
            ax.imshow(canvas_gt[i*J+j].detach().cpu().permute(1,2,0).numpy())
    fig.savefig(f"outputs/canvas_gt_batch.png")
    plt.close(fig)
    print(f"Saved canvas_gt_batch.png")
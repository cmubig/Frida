import os
os.environ['QT_QPA_PLATFORM']='offscreen' # Helps with matplotlib on some systems. Disables plotting on screen 

import sys
import torch 
import argparse
import datetime
from matplotlib import pyplot as plt

from options import Options
from my_tensorboard import TensorBoard
from brush_stroke import BrushStrokeBatch
from painting import PaintingBatch

###################
# Hyperparameters #
###################
device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
saving = True
plotting = False
batch_size = 6
total_size = 6
n_strokes = 50
dataset_path = f"stroke_generator/datasets/rand_{n_strokes}stroke_{total_size}.pth"

if __name__ == '__main__':
    opt = Options()
    # Initialize opt to set some in code
    opt.parser = opt.initialize(argparse.ArgumentParser(description="FRIDA Robot Painter"))
    opt.CANVAS_WIDTH_M = 0.18
    opt.CANVAS_HEIGHT_M = opt.CANVAS_WIDTH_M
    opt.gather_options()

    # Setup logging
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)

    # Setup dims
    h_render = int(opt.render_height)
    w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

    all_canvases = torch.ones(total_size,n_strokes+1,3,h_render,w_render).detach().to(device)
    all_palletes = torch.rand(total_size,12,3).detach().to(device)
    all_strokes = torch.zeros(total_size, n_strokes, 9).detach().to(device)
    total_count = 0
    while total_count < total_size:
        with torch.no_grad():
            # Setup canvases
            canvases = torch.ones(batch_size,n_strokes+1,3,h_render,w_render).detach().to(device)
            color_pallete = all_palletes[total_count:total_count+batch_size]

            # Generate a batch of random brush strokes
            strokes_tensor = torch.zeros(batch_size, n_strokes, 9).detach().to(device)
            brush_stroke_batches = []
            for i in range(n_strokes):
                stroke = BrushStrokeBatch(opt, batch_size, ink=None, init_differentiably=False).to(device)
                rand_color_choice = torch.randint(0,12,(batch_size,))
                stroke.color_transform = torch.nn.Parameter(color_pallete[torch.arange(batch_size),rand_color_choice])
                brush_stroke_batches.append(stroke)
                strokes_tensor[:,i] = torch.cat([
                    stroke.stroke_length,
                    stroke.stroke_z,
                    stroke.stroke_bend,
                    stroke.transformation.a,
                    stroke.transformation.xt,
                    stroke.transformation.yt,
                    stroke.color_transform
                ], dim=1)
                # Initialize painting and paint stroke
                painting = PaintingBatch(opt, background_img=canvases[:,i]).to(device)
                canvases[:,i+1] = painting([stroke], h_render, w_render, use_alpha=False, return_alphas=False)
        
        # Save batch
        all_canvases[total_count:total_count+batch_size] = canvases
        all_strokes[total_count:total_count+batch_size] = strokes_tensor
        total_count += batch_size
        print(f"Generated {total_count}/{total_size} samples")

    if saving:
        # Save model inputs as tensordict
        data = {
            'canvases': all_canvases,
            'palletes': all_palletes,
            'strokes': all_strokes,
        }
        torch.save(data, dataset_path)

    if plotting:
        # Plot and save canvases
        fig, axs = plt.subplots(4,4)
        i = 0
        I,J = axs.shape
        for i in range(I):
            for j in range(J):
                ax = axs[i,j]
                ax.imshow(canvases[i*J+j,-1].detach().cpu().permute(1,2,0).numpy())
        fig.savefig(f"outputs/canvas_gt_batch.png")
        plt.close(fig)
        print(f"Saved canvas_gt_batch.png")
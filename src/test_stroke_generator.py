import os
os.environ['QT_QPA_PLATFORM']='offscreen' # Helps with matplotlib on some systems. Disables plotting on screen 

import sys
import clip 
import torch
import argparse
import datetime
from matplotlib import pyplot as plt

from options import Options
from my_tensorboard import TensorBoard
from brush_stroke import BrushStrokeBatch
from painting import PaintingBatch
from stroke_generator.IL.model import StrokePredictor

###################
# Hyperparameters #
###################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8 # Make divisible by 4 for plotting
n_strokes = 1

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
    save_folder = f"outputs/test_{run_name}"
    os.mkdir(save_folder)

    # Setup dims
    h_render = int(opt.render_height)
    w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

    # Model and optimizer
    model = StrokePredictor(opt, device)
    model_path = "outputs/run_03_25__15_39_12_best/stroke_generator_state_dict_03_25__15_39_12.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

    # Setup canvases
    blank_canvas = torch.ones(batch_size,3,h_render,w_render).to(device)

    # Generate a batch of random brush strokes
    strokes_tensor = torch.zeros(batch_size, n_strokes, 9).to(device)
    strokes_left_tensor = torch.zeros(batch_size, n_strokes, 1).to(device)
    brush_stroke_batches = []
    for i in range(n_strokes):
        stroke = BrushStrokeBatch(opt, batch_size, ink=None, init_differentiably=False).to(device)
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
        strokes_left_tensor[:,i] = torch.ones(batch_size,i+1).to(device)
    
    # Initialize painting and paint brush stroke[s]
    painting = PaintingBatch(opt, background_img=blank_canvas).to(device)
    canvas_gt = painting(brush_stroke_batches, h_render, w_render, use_alpha=False, return_alphas=False)

    # Prepare inputs
    tokenized_text = clip.tokenize(["A painting"]*batch_size).to(device)
    color_palette = -1*torch.ones(blank_canvas.shape[0], 12, 3).to(device) # -1 is a special value for no color

    # Forward pass
    strokes_pred = model(
        current_canvas=blank_canvas,
        target_img=canvas_gt,
        target_tokenized_txt=tokenized_text, # Make sure this matches in edge case of leftover batch piece
        remaining_strokes=strokes_left_tensor[:,0],
        color_palette=color_palette
    )
    strokes_pred = strokes_pred.unsqueeze(-1)
    pred_brush_stroke_batches = []
    for i in range(n_strokes):
        # Model outputs [l, z, b, a, x, y, r, g, b]
        generated_stroke = BrushStrokeBatch(opt,
                                    stroke_length=strokes_pred[:,0],
                                    stroke_z=strokes_pred[:,1],
                                    stroke_bend=strokes_pred[:,2],
                                    stroke_alpha=torch.zeros(batch_size,1).to(device),
                                    color=strokes_pred[:,6:].squeeze(-1),
                                    a=strokes_pred[:,3],
                                    xt=strokes_pred[:,4],
                                    yt=strokes_pred[:,5],
                                    init_differentiably=True,
                                    ink=None)
        pred_brush_stroke_batches.append(generated_stroke)

    generated_painting = PaintingBatch(opt, background_img=blank_canvas).to(device)
    generated_canvas = generated_painting(pred_brush_stroke_batches, h_render, w_render, use_alpha=False, return_alphas=False)
    
    joint_canvas = torch.cat([canvas_gt, torch.zeros((batch_size,3,canvas_gt.shape[-2],3)).to(device), generated_canvas], dim=-1)

    # Plot and save canvases
    for i in range(batch_size):
        plt.imshow(joint_canvas[i].detach().cpu().permute(1,2,0).numpy())
        plt.savefig(os.path.join(save_folder,f"test_{run_name}_{i}.png"))
        plt.close()
    
    print(f"Saved test_canvas_{run_name}.png")
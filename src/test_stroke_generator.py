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
from stroke_generator.model import StrokePredictor
from stroke_generator.SAC.random_expert import RandomExpert

###################
# Hyperparameters #
###################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8 # Make divisible by 4 for plotting
n_strokes = 3
model_path = "outputs/run_04_21__00_22_07/3_Online_IL/stroke_generator_state_dict_04_21__00_22_07.pth"

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

    # Disable gradient tracking
    torch.no_grad() 

    # Model and expert
    random_expert = RandomExpert(opt, device)
    model = StrokePredictor(opt, device)
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")

    # Setup canvases
    starting_canvas = torch.ones(batch_size,3,h_render,w_render).to(device)
    color_palette = torch.rand(batch_size, 12, 3).to(device)
    remaining_strokes = n_strokes*torch.ones(batch_size,1).to(device)
    gt_strokes, gt_canvases = random_expert.rollout_trajectory(remaining_strokes.clone(), starting_canvas.clone(), color_palette.clone())
    gt_strokes = gt_strokes.detach().clone()
    target_img = gt_canvases[:,-1].clone()
    current_canvas = gt_canvases[:,0].clone()
    
    tokenized_text = clip.tokenize(["A splash of colors on a white background"]*batch_size).detach().to(device)
    mask = torch.ones(batch_size, 1).to(device)
    
    model.eval()
    model.save_hx = True
    for i in range(n_strokes):
        # Generate stroke
        stroke_tensor = model.sample(
            current_canvas=current_canvas,
            target_img=target_img,
            target_tokenized_txt=tokenized_text,
            remaining_strokes=remaining_strokes,
            color_palette=color_palette,
            mask=mask,
        )

        # Paint stroke
        stroke_tensor = stroke_tensor.unsqueeze(-1)
        generated_stroke = BrushStrokeBatch(opt,
                                    stroke_length=stroke_tensor[:,0],
                                    stroke_z=stroke_tensor[:,1],
                                    stroke_bend=stroke_tensor[:,2],
                                    stroke_alpha=torch.zeros(batch_size,1).to(device),
                                    color=stroke_tensor[:,6:].squeeze(-1),
                                    a=stroke_tensor[:,3],
                                    xt=stroke_tensor[:,4],
                                    yt=stroke_tensor[:,5],
                                    init_differentiably=True,
                                    ink=None)
        generated_painting = PaintingBatch(opt, background_img=current_canvas).to(device)
        generated_canvas = generated_painting([generated_stroke], h_render, w_render, use_alpha=False, return_alphas=False)

        # Update canvas
        current_canvas = generated_canvas
        remaining_strokes -= 1
        mask = torch.where(remaining_strokes <= 0, torch.zeros_like(mask), torch.ones_like(mask)).to(torch.float)
    
    # Add target canvas to generated canvas and save
    output_canvas = torch.cat(
         [target_img,
          torch.zeros((batch_size,3,target_img.shape[-2],3)).to(device), 
          generated_canvas], dim=-1)
    
    # Plot and save canvases
    for i in range(batch_size):
        name = f"test_canvas_{run_name}_{i}.png"
        plt.imshow(output_canvas[i].detach().cpu().permute(1,2,0).numpy())
        plt.savefig(os.path.join(save_folder,name))
        plt.close()
        print(f"Saved {name}")
    
import os
os.environ['QT_QPA_PLATFORM']='offscreen' # Helps with matplotlib on some systems. Disables plotting on screen 

import sys
import torch
import argparse
import datetime

from options import Options
from my_tensorboard import TensorBoard
from stroke_generator.model import StrokePredictor

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
    save_folder = f"outputs/model_spliced_state_dict_{run_name}"
    os.mkdir(save_folder)

    # Setup dims
    h_render = int(opt.render_height)
    w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

    # Model and optimizer
    model = StrokePredictor(opt, device)
    
    split_path = "outputs/model_split_state_dicts_03_31__20_36_18"

    try:
        model.state_encoder.load_state_dict(torch.load(os.path.join(split_path,"state_encoder_state_dict.pth")))
        print(f"Loaded state encoder")
    except:
        print(f"Couldn't load state encoder")
    
    try:
        model.main.load_state_dict(torch.load(os.path.join(split_path,"main_state_dict.pth")))
        print(f"Loaded main")
    except:
        print(f"Couldn't load main")
    
    try:
        model.stroke_decoder.dec_mu_l.load_state_dict(torch.load(os.path.join(split_path,"dec_l_state_dict.pth")))
        model.stroke_decoder.dec_mu_z.load_state_dict(torch.load(os.path.join(split_path,"dec_z_state_dict.pth")))
        model.stroke_decoder.dec_mu_b.load_state_dict(torch.load(os.path.join(split_path,"dec_b_state_dict.pth")))
        model.stroke_decoder.dec_mu_a.load_state_dict(torch.load(os.path.join(split_path,"dec_a_state_dict.pth")))
        model.stroke_decoder.dec_mu_xy.load_state_dict(torch.load(os.path.join(split_path,"dec_xy_state_dict.pth")))
        model.stroke_decoder.dec_rgb.load_state_dict(torch.load(os.path.join(split_path,"dec_rgb_state_dict.pth")))
        print(f"Loaded decoders")
    except:
        print(f"Couldn't load decoders")

    # Save full model
    torch.save(model.state_dict(), os.path.join(save_folder,f"full_model_state_dict.pth"))
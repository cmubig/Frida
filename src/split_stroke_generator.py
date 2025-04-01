import os
os.environ['QT_QPA_PLATFORM']='offscreen' # Helps with matplotlib on some systems. Disables plotting on screen 

import sys
import torch
import argparse
import datetime

from options import Options
from my_tensorboard import TensorBoard
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
    save_folder = f"outputs/model_split_state_dicts_{run_name}"
    os.mkdir(save_folder)

    # Setup dims
    h_render = int(opt.render_height)
    w_render = int(opt.render_height*(opt.CANVAS_WIDTH_M/opt.CANVAS_HEIGHT_M))

    # Model and optimizer
    model = StrokePredictor(opt, device)
    model_path = "outputs/run_03_27__19_34_20/stroke_generator_state_dict_03_27__19_34_20_p1.pth"
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded model from {model_path}")


    torch.save(model.state_encoder.state_dict(), os.path.join(save_folder,f"state_encoder_state_dict.pth"))
    torch.save(model.main.state_dict(), os.path.join(save_folder,f"main_state_dict.pth"))
    torch.save(model.dec_l.state_dict(), os.path.join(save_folder,f"dec_l_state_dict.pth"))
    torch.save(model.dec_z.state_dict(), os.path.join(save_folder,f"dec_z_state_dict.pth"))
    torch.save(model.dec_b.state_dict(), os.path.join(save_folder,f"dec_b_state_dict.pth"))
    torch.save(model.dec_a.state_dict(), os.path.join(save_folder,f"dec_a_state_dict.pth"))
    torch.save(model.dec_xy.state_dict(), os.path.join(save_folder,f"dec_xy_state_dict.pth"))
    torch.save(model.dec_rgb.state_dict(), os.path.join(save_folder,f"dec_rgb_state_dict.pth"))

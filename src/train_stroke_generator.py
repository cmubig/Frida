import os
# Set some env vars to help run smoothly
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True' # In theory helps with torch memory allocation
os.environ['QT_QPA_PLATFORM']='offscreen' # Helps with matplotlib on some systems. Disables plotting on screen 
import sys
import datetime
import argparse

import torch
from matplotlib import pyplot as plt

from options import Options
from my_tensorboard import TensorBoard

from stroke_generator.IL.model import StrokePredictor
from stroke_generator.utils.canvas_dataset import CanvasDataset, SequentialCanvasDataset
from stroke_generator.trainer_IL import OfflineStrokeGeneratorTrainer, OfflineMultiStrokeGeneratorTrainer

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
    save_folder = f"outputs/run_{run_name}"
    os.mkdir(save_folder)

    # Model and optimizer
    model = StrokePredictor(opt, device)
    model_path = "outputs/run_03_26__14_48_03/stroke_generator_state_dict_03_26__14_48_03_p3.pth"
    if model_path is not None and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        except:
            print(f"Couldn't load model from {model_path}")
    
    
    dataset = SequentialCanvasDataset(opt, "stroke_generator/datasets/rand_1stroke_1k_0.pth")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    offline_trainer = OfflineMultiStrokeGeneratorTrainer(
        opt,
        model,
        device,
        dataloader=dataloader,
        save_folder=save_folder,
        run_name=run_name+"_p1"
    )
    print("Beginning 1 stroke Training")
    offline_trainer.train(epochs=50)

    print("Adding 2nd stroke")
    dataset.load_data("stroke_generator/datasets/rand_2stroke_1k_0.pth")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    offline_trainer = OfflineMultiStrokeGeneratorTrainer(
        opt,
        model,
        device,
        dataloader=dataloader,
        save_folder=save_folder,
        run_name=run_name+"_p2"
    )
    print("Beginning 1-2 stroke Training")
    offline_trainer.train(epochs=50)

    print("Adding 3rd stroke")
    dataset.load_data("stroke_generator/datasets/rand_3stroke_1k_0.pth")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    offline_trainer = OfflineMultiStrokeGeneratorTrainer(
        opt,
        model,
        device,
        dataloader=dataloader,
        save_folder=save_folder,
        run_name=run_name+"_p3"
    )
    print("Beginning 1-3 stroke Training")
    offline_trainer.train(epochs=100)
    
    print("Finished training")

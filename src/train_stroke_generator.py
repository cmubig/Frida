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

from stroke_generator.model import StrokePredictor, DeterministicStrokePredictor
from stroke_generator.utils.canvas_dataset import SequentialCanvasDataset
from stroke_generator.trainers.online_IL import OnlineMultiStrokeGeneratorTrainer
from stroke_generator.trainers.trainer_SAC import SACTrainer

from stroke_generator.trainers.trainer_DDPG import DDPGTrainer
from stroke_generator.SAC.random_expert import RandomExpert, GridExpert
from stroke_generator.env import StrokeEnv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_canvas_as_img(canvas, path='outputs', suffix=''):
    plt.imshow(canvas.detach().cpu().permute(1,2,0).numpy())
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            raise Exception(f"Path provided for saving canvas does not exist and couldn't be created.\n\t{path}")
    plt.savefig(os.path.join(path,f"canvas{suffix}.png"))

def train(trainer, run_name="run_data", epochs=100):
    print(f"Beginning training for {run_name} for {epochs} epochs")
    subfolder = os.path.join(save_folder, run_name)
    os.mkdir(subfolder)
    return trainer.train(epochs)

if __name__ == '__main__':
    opt = Options()
    # Initialize opt to set some in code
    opt.parser = opt.initialize(argparse.ArgumentParser(description="FRIDA Robot Painter"))
    opt.CANVAS_WIDTH_M = 0.18
    opt.CANVAS_HEIGHT_M = opt.CANVAS_WIDTH_M # * 0.667
    opt.gather_options()

    # Setup logging
    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)
    save_folder = f"outputs/run_{run_name}"
    os.mkdir(save_folder)

    # Model and optimizer
    model = DeterministicStrokePredictor(opt, device)
    model_path = None
    if model_path is not None and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model from {model_path}")
        except:
            print(f"Couldn't load model from {model_path}")

    # Make trainer
    sub_save_folder = os.path.join(save_folder, "DDPG_pretrain")
    if not os.path.exists(sub_save_folder):
        os.mkdir(sub_save_folder)
    trainer = DDPGTrainer(
        opt,
        model,
        device=device,
        batch_size=256,
        buffer_size=5_000,
        save_folder=sub_save_folder
    )
    checkpoint_path = None # "outputs/run_06_16__15_46_39/DDPG_25_stroke/checkpoint.pth"
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        trainer.actor.load_state_dict(checkpoint['actor_state_dict'])
        trainer.critic.load_state_dict(checkpoint['critic_state_dict'])
        trainer.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        trainer.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if hasattr(trainer, 'replay_buffer') and 'replay_buffer' in checkpoint:
            trainer.replay_buffer = checkpoint['replay_buffer']
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    # print("############################")
    # print(" Phase 1: One Stroke Training")
    # print("############################")
    # expert = GridExpert(opt, device, grid_size=1, noise_level=0.001)
    # env = StrokeEnv(opt, device, num_envs=256, target_expert=expert, max_strokes=1)
    # trainer.train(env, 
    #               num_episodes=500, 
    #               noise_start=0.3, 
    #               min_noise=0.01, 
    #               noise_decay=0.99,
    #               checkpoint_interval=50)

    # Make the problem a little harder by increasing the grid size
    # sub_save_folder = os.path.join(save_folder, "DDPG_1_stroke")
    # if not os.path.exists(sub_save_folder):
    #     os.mkdir(sub_save_folder)
    # trainer.save_folder = sub_save_folder
    # expert = GridExpert(opt, device, grid_size=2, noise_level=0.005)
    # env = StrokeEnv(opt, device, num_envs=256, target_expert=expert, max_strokes=1)
    # trainer.train(env, num_episodes=100, noise_start=0.2, min_noise=0.01, noise_decay=0.99)

    # # Add a 3rd stroke
    # print("############################")
    # print(" Phase 3: Three Stroke Training")
    # print("############################")
    # sub_save_folder = os.path.join(save_folder, "DDPG_3_stroke")
    # if not os.path.exists(sub_save_folder):
    #     os.mkdir(sub_save_folder)
    # trainer.save_folder = sub_save_folder
    # expert = GridExpert(opt, device, grid_size=4, noise_level=0.01)
    # env = StrokeEnv(opt, device, num_envs=256, target_expert=expert, max_strokes=3)
    # trainer.train(env, num_episodes=2000, noise_start=0.2, min_noise=0.01, noise_decay=0.99)

    # Add a 5th stroke
    print("############################")
    print(" Phase 4: Five Stroke Training")
    print("############################")
    sub_save_folder = os.path.join(save_folder, "DDPG_5_stroke")
    if not os.path.exists(sub_save_folder):
        os.mkdir(sub_save_folder)
    trainer.save_folder = sub_save_folder
    expert = GridExpert(opt, device, grid_size=5, noise_level=0.01)
    env = StrokeEnv(opt, device, num_envs=128, target_expert=expert, max_strokes=5)
    trainer.train(env, 
                  num_episodes=20000, 
                  noise_start=0.1, 
                  min_noise=0.01, 
                  noise_decay=0.99,
                  checkpoint_interval=20)

    # # Add up to 10 strokes
    # print("############################")
    # print(" Phase 5: Ten Stroke Training")
    # print("############################")
    # sub_save_folder = os.path.join(save_folder, "DDPG_10_stroke")
    # if not os.path.exists(sub_save_folder):
    #     os.mkdir(sub_save_folder)
    # trainer.save_folder = sub_save_folder
    # expert = GridExpert(opt, device, grid_size=10, noise_level=0.1)
    # env = StrokeEnv(opt, device, num_envs=256, target_expert=expert, max_strokes=10)
    # trainer.train(env, 
    #               num_episodes=5000, 
    #               noise_start=0.3, 
    #               min_noise=0.01, 
    #               noise_decay=0.99,
    #               checkpoint_interval=100)
    
    print("############################")
    print(" Phase 7: 25 Stroke Training")
    print("############################")
    sub_save_folder = os.path.join(save_folder, "DDPG_25_stroke")
    if not os.path.exists(sub_save_folder):
        os.mkdir(sub_save_folder)
    trainer.save_folder = sub_save_folder
    expert = GridExpert(opt, device, grid_size=10, noise_level=0.1)
    env = StrokeEnv(opt, device, num_envs=64, target_expert=expert, max_strokes=25)
    trainer.train(env, 
                  num_episodes=1_000_000, 
                  noise_start=0.2, 
                  min_noise=0.01, 
                  noise_decay=0.995,
                  checkpoint_interval=10)
    """
    print("############################")
    print(" Phase 1: Low Stroke Count")
    print("############################")
    print()

    dataset = SequentialCanvasDataset(opt, f"stroke_generator/datasets/rand_50stroke_6.pth")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False)
    offline_trainer = OfflineMultiStrokeGeneratorTrainer(
        opt,
        model,
        device,
        dataloader=dataloader,
        save_folder=subfolder,
        run_name=run_name
    )
    model = train(offline_trainer, run_name="pretraining", epochs=5000)

    stroke_count = [1,3,5,7]
    epoch_count = [10, 20, 50, 100]
    dataset = SequentialCanvasDataset(opt, f"stroke_generator/datasets/rand_{stroke_count[0]}stroke_500.pth")
    for i,s in enumerate(stroke_count):
        subfolder = os.path.join(save_folder, f"")
        os.mkdir(subfolder)
        if s != stroke_count[0]:
            print(f"Adding {s} stroke data to datset")
            dataset.load_data(f"stroke_generator/datasets/rand_{s}stroke_500.pth")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        offline_trainer = OfflineMultiStrokeGeneratorTrainer(
            opt,
            model,
            device,
            dataloader=dataloader,
            save_folder=subfolder,
            run_name=run_name
        )
        model = train(offline_trainer, "{s}stroke", epochs=epoch_count[i])

    del offline_trainer
    del dataset
    del dataloader
    

    print("############################")
    print(" Phase 2: High Stroke Count")
    print("############################")

    print()
    print("Swapping to 9 and 11 stroke dataset")

    print("Loading 11 stroke dataset")
    dataset = SequentialCanvasDataset(opt, f"stroke_generator/datasets/rand_11stroke_500.pth")
    print("Loading 9 stroke dataset")
    dataset.load_data(f"stroke_generator/datasets/rand_9stroke_500.pth")
    subfolder = os.path.join(save_folder, f"9and11stroke")
    os.mkdir(subfolder)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    offline_trainer = OfflineMultiStrokeGeneratorTrainer(
        opt,
        model,
        device,
        dataloader=dataloader,
        save_folder=subfolder,
        run_name=run_name
    )
    print(f"Beginning 9 and 11 stroke Training")
    model = offline_trainer.train(epochs=500)

    del offline_trainer
    del dataset
    del dataloader

    

    # print("############################")
    # print(" Phase 3: Online IL")
    # print("############################")
    # subfolder = os.path.join(save_folder, f"3_Online_IL_1")
    # os.mkdir(subfolder)
    # trainer = OnlineMultiStrokeGeneratorTrainer(
    #     opt,
    #     model,
    #     device,
    #     batch_size=512,
    #     max_strokes=1,
    #     save_folder=subfolder,
    #     run_name=run_name
    # )
    # model = trainer.train(epochs=5000)

    # subfolder = os.path.join(save_folder, f"3_Online_IL_2")
    # os.mkdir(subfolder)
    # trainer = OnlineMultiStrokeGeneratorTrainer(
    #     opt,
    #     model,
    #     device,
    #     batch_size=256,
    #     max_strokes=2,
    #     save_folder=subfolder,
    #     run_name=run_name
    # )
    # model = trainer.train(epochs=5000)

    # subfolder = os.path.join(save_folder, f"3_Online_IL_3")
    # os.mkdir(subfolder)
    # trainer = OnlineMultiStrokeGeneratorTrainer(
    #     opt,
    #     model,
    #     device,
    #     batch_size=256,
    #     max_strokes=3,
    #     save_folder=subfolder,
    #     run_name=run_name
    # )
    # model = trainer.train(epochs=5000)

    # subfolder = os.path.join(save_folder, f"3_Online_IL_5")
    # os.mkdir(subfolder)
    # trainer = OnlineMultiStrokeGeneratorTrainer(
    #     opt,
    #     model,
    #     device,
    #     batch_size=128,
    #     max_strokes=5,
    #     save_folder=subfolder,
    #     run_name=run_name
    # )
    # model = trainer.train(epochs=4000)

    # subfolder = os.path.join(save_folder, f"3_Online_IL_7")
    # os.mkdir(subfolder)
    # trainer = OnlineMultiStrokeGeneratorTrainer(
    #     opt,
    #     model,
    #     device,
    #     batch_size=128,
    #     max_strokes=7,
    #     save_folder=subfolder,
    #     run_name=run_name
    # )
    # model = trainer.train(epochs=4000)

    # subfolder = os.path.join(save_folder, f"3_Online_IL_9")
    # os.mkdir(subfolder)
    # trainer = OnlineMultiStrokeGeneratorTrainer(
    #     opt,
    #     model,
    #     device,
    #     batch_size=64,
    #     max_strokes=9,
    #     save_folder=subfolder,
    #     run_name=run_name
    # )
    # model = trainer.train(epochs=8000)
    
    # print("############################")
    # print(" Phase 4: SAC on Random Strokes") os.path.join(save_folder, f"3_Online_IL_3")
    # os.mkdir(subfolder)
    # trainer = OnlineMultiStrokeGeneratorTrainer(
    #     opt,
    #     model,
    #     device,
    #     batch_size=256,
    #     max_strokes=3,
    #     save_folder=subfolder,
    #     run_name=run_name
    # )
    # model = trainer.train(epochs=5000)
    # print()
    # print("Switching to single stroke Soft Actor Critic")
    # subfolder = os.path.join(save_folder, f"SAC_1")
    # os.mkdir(subfolder)
    # sac_trainer = SACTrainer(opt, model, device, num_envs=256, buffer_size=5000, save_folder=subfolder)
    # sac_trainer.max_strokes = 1
    # sac_trainer.seed_rollouts = 100
    # sac_trainer.update_freq = 50
    # sac_trainer.generate_new_target_freq = 10
    # sac_trainer.agent.critic.load_state_dict(torch.load("outputs/critic.pth"))
    # model = sac_trainer.train(rollouts=10000)
    # critic = sac_trainer.agent.critic
    # log_alpha = sac_trainer.agent.log_alpha

    # print()
    # print("Switching to 2 strokes")
    # subfolder = os.path.join(save_folder, f"SAC_2")
    # os.mkdir(subfolder)
    # sac_trainer = SACTrainer(opt, model, device, num_envs=256, buffer_size=5000, save_folder=subfolder)
    # sac_trainer.max_strokes = 2
    # sac_trainer.seed_rollouts = 150
    # sac_trainer.update_freq = 15
    # sac_trainer.agent.critic.load_state_dict(critic.state_dict())
    # sac_trainer.agent.log_alpha = log_alpha
    # del critic
    # del log_alpha
    # model = sac_trainer.train(rollouts=10000)
    
    print("Finished training")
    
    """
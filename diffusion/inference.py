import torch
import torchvision
import argparse
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from dataclasses import dataclass

from model import Unet
from noise import LinearNoiseScheduler
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((train_config.num_samples,
                      model_config.im_channels,
                      model_config.im_size,
                      model_config.im_size)).to(device)
    for i in tqdm(reversed(range(diffusion_config.num_timesteps))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))

        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # Save only the final image
    ims = torch.clamp(xt, -1., 1.).detach().cpu()
    ims = (ims + 1) / 2
    grid = make_grid(ims, nrow=train_config.num_grid_rows)
    img = torchvision.transforms.ToPILImage()(grid)
    if not os.path.exists(os.path.join(train_config.task_name, 'samples')):
        os.mkdir(os.path.join(train_config.task_name, 'samples'))
    img.save(os.path.join(train_config.task_name, 'samples', 'final_sample.png'))
    img.close()


def infer(args):
    diffusion_config = DiffusionConfig()
    model_config = UnetConfig()
    train_config = TrainConfig()

    # Load model with checkpoint
    model = Unet().to(device)
    model.load_state_dict(torch.load(os.path.join(train_config.task_name,
                                                  train_config.ckpt_name), map_location=device))
    model.eval()

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=diffusion_config.num_timesteps,
                                     beta_start=diffusion_config.beta_start,
                                     beta_end=diffusion_config.beta_end)
    scheduler.to(device)
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    args = parser.parse_args()
    infer(args)
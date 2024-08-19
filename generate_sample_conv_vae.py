import torch
from torchvision.utils import save_image, make_grid
import numpy as np
import os

from model.conv_vae import ConvVAE, vae_loss

# hyperparameters
latent_dim = 512
image_size = 128
in_channels = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load the model
model = ConvVAE(in_channels=in_channels, latent_dim=latent_dim).to(device)

# load weights
model.load_state_dict(torch.load('weights/conv_vae/final_checkpoint.pth'))  # Replace X with the epoch number

model.eval()

# main loop
num_samples = 32  # number of images to be generated
with torch.no_grad():
    for i in range(num_samples):
        # generate random latent vectors
        z = torch.randn(1, latent_dim).to(device)
    
        # decode the random latent vector
        generated_image = model.decode(z)
        generated_image = generated_image.to('cpu')

        save_image(generated_image, os.path.join('generated_samples/conv_vae/',
                                                 f'conv_vae_generated_sample_{i+1}.png'))
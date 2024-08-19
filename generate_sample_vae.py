import torch
from torchvision.utils import save_image, make_grid
import numpy as np
import os

from model.vae import VAE

# hyperparameters
latent_dim = 512
image_size = 128

# ensure the use of GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load the model
model = VAE(input_dim= 128*128,
            latent_dim=latent_dim).to(device)

# load weights
model.load_state_dict(torch.load('/weights/vae/final_checkpoint.pth'))

model.eval()

# main loop
num_samples = 32  # number of images to be generated
with torch.no_grad():
    for i in range(num_samples):
        # generate random latent vectors
        z = torch.randn(1, latent_dim).to(device)

        # decode the random latent vector
        generated_image = model.decode(z)
        generated_image = generated_image.view(1, 128, 128).to('cpu')

        save_image(generated_image, os.path.join('generated_samples/vae/',
                                                 f'vae_generated_sample_{i+1}.png'))
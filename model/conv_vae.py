import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super(ConvVAE, self).__init__()
        
        # encoder layers
        self.enc_conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1)  # 64x64
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 32x32
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # 16x16
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 8x8
        self.enc_conv5 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # 4x4
        self.enc_conv6 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)  # 2x2
        self.mu_layer = nn.Linear(1024 * 2 * 2, latent_dim)  # Latent space mean
        self.logvar_layer = nn.Linear(1024 * 2 * 2, latent_dim)  # Latent space log-variance
        
        # decoder layers
        self.fc_dec = nn.Linear(latent_dim, 1024 * 2 * 2)
        self.dec_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)  # 4x4
        self.dec_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # 8x8
        self.dec_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # 16x16
        self.dec_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 32x32
        self.dec_conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 64x64
        self.dec_conv6 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1)  # 128x128
    
    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = F.relu(self.enc_conv3(h))
        h = F.relu(self.enc_conv4(h))
        h = F.relu(self.enc_conv5(h))
        h = F.relu(self.enc_conv6(h))
        h = h.view(h.size(0), -1)  # flatten
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc_dec(z))
        h = h.view(h.size(0), 1024, 2, 2)
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = F.relu(self.dec_conv3(h))
        h = F.relu(self.dec_conv4(h))
        h = F.relu(self.dec_conv5(h))
        return torch.sigmoid(self.dec_conv6(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded_z = self.decode(z)
        return decoded_z, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return MSE + KLD

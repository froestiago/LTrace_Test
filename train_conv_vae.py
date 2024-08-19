import torch
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import get_dataloader
from model.conv_vae import ConvVAE, vae_loss

if __name__ == '__main__':

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # set train dataset path
    train_dataset_path = "train_images.h5"

    # setting hyperparameters
    in_channels = 1
    image_size = 128
    latent_dim = 512

    batch_size = 128
    num_epochs = 100
    learning_rate = 1e-3

    # defining dataset, model and optimizer
    train_loader = get_dataloader(file_path= train_dataset_path,
                                  batch_size= batch_size,
                                  num_workers= 16,
                                  transform= False)

    model = ConvVAE(in_channels= in_channels,
                latent_dim= latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # training loop
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        mse_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)


                # forward pass
                recon_batch, mu, logvar = model(data)

                # compute loss
                loss = vae_loss(recon_batch, data, mu, logvar)
                mse_batch = F.mse_loss(recon_batch, data, reduction='sum').item() / batch_size

                # backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                mse_loss += mse_batch
                optimizer.step()

                # update tqdm progress bar
                pbar.set_postfix({'loss': loss.item(), 'mse': mse_batch})
                pbar.update(1)

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_mse = mse_loss / len(train_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}, Training MSE: {avg_train_mse:.4f}')

    # save the model checkpoint
    torch.save(model.state_dict(), f'weights/conv_vae/final_checkpoint.pth')

    print("Training finished!")
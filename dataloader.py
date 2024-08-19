import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, file_path, split = 'full_train', transform= True):
        self.file_path = file_path
        self.transform = transform
        self.split = split
        
        self.transform_pipeline = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
            # transforms.ToTensor()
        ])
        
        if self.split == 'full_train':
            # Open the h5 file
            self.h5_file = h5py.File(file_path, 'r')
            self.dataset_name = list(self.h5_file.keys())[0]
            self.dataset = self.h5_file[self.dataset_name]

        if self.split == 'numpy_train' or self.split == 'numpy_val':
            self.dataset = np.load(file_path)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        image = self.dataset[idx]
        # (C, H, W)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        
        if self.transform:
            image = self.transform_pipeline(image)
        
        return image

# helper funtion
def get_dataloader(file_path, batch_size=32, split='full_train', num_workers=0, transform = True, shuffle = True):
    dataset = TrainDataset(file_path= file_path, split=split,transform= transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle= shuffle, num_workers=num_workers)
    return dataloader

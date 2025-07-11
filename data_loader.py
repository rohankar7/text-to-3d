import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
shuffle_condition = False
import config

class VoxelDataset(Dataset):
    def __init__(self, voxel_paths):
        self.voxel_paths = voxel_paths
    def __len__(self): return len(self.voxel_paths)
    def __getitem__(self, index):
        file_path = self.voxel_paths[index]
        voxel_data = torch.load(f"{file_path}", weights_only=False)
        assert voxel_data.shape == torch.Size([1, 64, 64, 64]), f"Unexpected shape: {voxel_data.shape}"
        return voxel_data
def voxel_dataloader():
    voxel_paths = [os.path.join(config.voxel_dir, path) for path in os.listdir(config.voxel_dir) if path.endswith(".pt")]
    dataset = VoxelDataset(voxel_paths)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=config.voxel_batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.voxel_batch_size, shuffle=False)
    return train_loader, test_loader
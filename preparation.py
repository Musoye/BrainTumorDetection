import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_data_loader(addr, batch_size=4):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize()
    ])

    load_data = ImageFolder(root=addr, transform=data_transform)
    print(load_data.class_to_idx)

    train_size = int(0.8 * len(load_data))
    val_size = len(load_data) - train_size
    train_dataset, val_dataset = random_split(load_data, [train_size, val_size])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

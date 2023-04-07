import torchvision.transforms as transforms
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
from src.data.dataset import ImageDataset
from src.enities.training_params import TrainingParams


def get_loaders(params: TrainingParams):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(params.img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ToTensor(),
    ])
    train_csv = pd.read_csv(os.path.join(params.path_to_dataset, 'train_.csv'))
    train_data = ImageDataset(
        params.path_to_dataset, train_csv, transform=transform_train
    )
    transform_valid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(params.img_size),
        transforms.ToTensor(),
    ])
    valid_csv = pd.read_csv(os.path.join(params.path_to_dataset, 'valid_.csv'))
    valid_data = ImageDataset(
        params.path_to_dataset, valid_csv, transform=transform_valid
    )
    train_loader = DataLoader(
        train_data,
        batch_size=params.batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=params.batch_size,
        shuffle=False
    )
    return train_data, train_loader, valid_data, valid_loader

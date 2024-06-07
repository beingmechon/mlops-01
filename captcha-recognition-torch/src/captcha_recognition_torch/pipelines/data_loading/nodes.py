import os

from torch.utils.data import  DataLoader
from .captcha_dataset import CAPTCHADataset


def get_dataset(data_path):
    return CAPTCHADataset(os.path.join(data_path))

def get_dataloader(train_folder, test_folder, batch_size, shuffle):
    train_loader = DataLoader(get_dataset(train_folder), batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(get_dataset(test_folder), batch_size=batch_size, shuffle=shuffle)

    print("Train and Test Data Loader Object created.")
    return [train_loader, test_loader]
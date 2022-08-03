from torch.utils.data import Dataset, random_split
from torch.nn.functional import normalize
import torch
import pandas as pd
import numpy as np


class HEPDataset(Dataset):
    def __init__(self, csv_file, transform=None) -> None:
        self.events = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        sample = np.array(self.events.iloc[index, 1:])
        sample = sample.astype('float')
        
        if self.transform:
            sample = self.transform(sample)

        return sample


def split(dataset):
    total_count = len(dataset)
    train_count = int(0.75 * total_count)
    valid_count = int(0.25 * total_count)
    # test_count = total_count - train_count - valid_count
    train_dataset, valid_dataset = random_split(dataset, (train_count, valid_count))
    return train_dataset, valid_dataset

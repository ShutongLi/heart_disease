import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, dataframe):

        X, y = dataframe.drop('y', axis = 1).values, dataframe['y'].values

        self.x_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

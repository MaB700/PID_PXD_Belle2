import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CreateCNNDataset(Dataset):

    def __init__(self, path, n_samples, label):
        data = pd.read_csv(path, header=None, delimiter= " ", nrows=n_samples).values.astype('float64')
        x = torch.from_numpy(data[:, 2:83]/255.0)
        y = torch.tensor([np.double(label)]) 

        self.x = torch.reshape(x, (n_samples, 1, 9, 9))
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
import numpy as np
import torch
import random
from torch.utils.data import Dataset


class GaussianDataSet(Dataset):
    def __init__(self, data_size, mean, stddev):
        self.data_size = data_size
        self.data = np.random.normal(mean, stddev, size=data_size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        return torch.tensor([self.data[index]], dtype=torch.float)



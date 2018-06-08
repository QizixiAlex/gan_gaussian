import torch.nn.functional as F
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu6(self.map1(x))
        x = F.relu6(self.batch_norm(self.map2(x)))
        x = F.sigmoid(self.map3(x))
        return x

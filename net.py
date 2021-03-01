import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from siren_pytorch import Siren, SirenNet
from layers import ResonanceModule, BlackWhite
import math


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)

        # 9216, 12544
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.act = torch.relu

    def forward(self, x):
        x = self.conv1(x)

        x = self.act(x)

        x = self.conv2(x)


        x = self.act(x)
        x = self.dropout1(x)
        # x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.act(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


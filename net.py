import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)

        # 9216, 12544
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout2d = nn.Dropout2d(0.5)
        self.dropout1d = nn.Dropout(0.5)

        self.act = torch.relu

    def forward(self, x):
        x = self.conv1(x)

        x = self.act(x)
        x = self.dropout2d(x)

        x = self.conv2(x)

        x = self.act(x)
        x = self.dropout2d(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = self.act(x)
        x = self.dropout1d(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
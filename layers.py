from siren_pytorch import Siren, SirenNet, Sine
import torch, torch.nn as nn, torch.nn.functional as F
import math
from copy import deepcopy
import numpy as np

import matplotlib.pyplot as plt

EPS = torch.finfo(torch.float32).eps

# https://github.com/dipuk0506/SpinalNet
class SpinalResNet(nn.Module):
    def __init__(self, in_features, hidden_size=128, out_features=10):
        super().__init__()
        half_in_size = round(in_features/2)
        self.half_in_size = half_in_size
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Linear(half_in_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Linear(half_in_size+hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Linear(half_in_size+hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Linear(half_in_size+hidden_size, hidden_size),
            # nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size*4, out_features),)
        
    def forward(self, x):
        half_in_size = self.half_in_size
        x1 = self.fc_spinal_layer1(x[:, 0:half_in_size])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,half_in_size:2*half_in_size], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:half_in_size], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,half_in_size:2*half_in_size], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)

        x = self.fc_out(x)
        return x

class ResonanceModule(nn.Module):
    def __init__(self, output_size, dist=torch.distributions.Uniform(-math.sqrt(3), math.sqrt(3)), custom_siren=None, t_dim=1, hidden=128):
        super().__init__()
        self.resonance_modules = []
        self.dist = dist
        self.t_dim = t_dim
        if custom_siren is None:
            # self.understanding = SirenNet(output_size, hidden, t_dim, 3, final_activation=None, w0=30)
            # self.understanding = torch.nn.Sequential(nn.Linear(output_size, hidden),
            #                                          nn.ReLU(),
            #                                          nn.Linear(hidden, hidden),
            #                                          nn.ReLU(),
                                                    #  nn.Linear(hidden, t_dim))
            self.understanding = SpinalResNet(output_size, hidden, t_dim)
        else:
            self.understanding = custom_siren
        self.understanding_loss = torch.nn.MSELoss()
    
    def self_understanding(self, outputs, t, detach=False):
        if detach:
            outputs = outputs.detach()
        # std, mean = torch.std_mean(outputs, dim=1, keepdim=True)
        # outputs = (outputs - mean) / std
        preds = self.understanding(outputs)
        return self.understanding_loss(preds, t)

    def pluck(self, size=1, device='cuda'):
        return self.dist.sample(sample_shape=(size, self.t_dim)).to(device)


class BlackWhite:
    def __init__(self, p=0.5):
        self.bern = torch.distributions.Bernoulli(probs=torch.Tensor([p]))

    def forward(self, x):
        b = self.bern.sample(x.size(), ).squeeze(-1).to('cuda')
        return b, 1-b

# class BlackWhite2d(nn.Module):
#     def __init__(self, p=0.5):
#         self.bern = torch.distributions.Bernoulli(probs=torch.Tensor[p])

#     def forward(self, x):
#         b = self.bern.sample(x.size())
#         b.unsqueeze(-1).unsqueeze(-1)
#         return b, 1-b
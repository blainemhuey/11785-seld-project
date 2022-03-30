import torch
import torch.nn as nn


class SELDNet(nn.Module):

    def __init__(self, in_channels, p, mp, r, n):
        super().__init__()

        base_layers = []
        for i in range(3):
            base_layers.append(nn.Conv2d(in_channels, p, 3, padding=1))
            base_layers.append(nn.ReLU())
            base_layers.append(nn.BatchNorm2d(p))
            base_layers.append(nn.MaxPool2d(kernel_size=(1, mp[i])))
            in_channels = p

        for i in range(2):
            base_layers.append(nn.GRU(input_size=p, hidden_size=128, bidirectional=True))
            base_layers.append(nn.Tanh())

        self.base = nn.Sequential(*base_layers)

        sed_layers = [nn.Linear(p, r), nn.Linear(r, n), nn.Sigmoid()]
        self.sed = nn.Sequential(*sed_layers)

        doa_layers = [nn.Linear(p, r), nn.Linear(r, 3*n), nn.Tanh()]
        self.doa = nn.Sequential(*doa_layers)

    def forward(self, x):
        base = self.base(x)
        return self.sed(base), self.doa(base)

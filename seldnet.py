import torch
import torch.nn as nn


class SELDNet(nn.Module):

    def __init__(self, in_channels=7, bins=64, c=13, p=None, mp=None):
        super().__init__()

        if mp is None:  # CNN Max Pooling Filter Size Per Layer
            mp = [(5, 4), (1, 4), (1, 2)]

        if p is None:  # TODO: Correct? Output CNN Channels Per Layer
            p = [16, 32, 64]

        base_layers = []
        for i in range(3):
            base_layers.append(nn.Conv2d(in_channels, p[i], 3, padding=1))
            base_layers.append(nn.ReLU())
            base_layers.append(nn.BatchNorm2d(bins))
            base_layers.append(nn.MaxPool2d(kernel_size=mp[i], stride=mp[i]))
            bins = bins // mp[i][1]

        for i in range(2):
            base_layers.append(nn.GRU(input_size=128, hidden_size=128, bidirectional=True)) # TODO: What is hidden size?
            base_layers.append(nn.Tanh())

        self.base = nn.Sequential(*base_layers)

        seld_layers = [nn.Linear(128, 128), nn.Linear(128, 3 * 3 * c), nn.Tanh()]
        self.seld = nn.Sequential(*seld_layers)

    def forward(self, x):
        base = self.base(x)
        _, t, _ = base.shape
        reshaped = base.reshape((t, 128))
        return self.seld(reshaped)

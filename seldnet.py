import torch
import torch.nn as nn


class SELDNet(nn.Module):

    def __init__(self, in_channels=7, gru_hidden_size=128, c=13, p=None, mp=None):
        super().__init__()

        if mp is None:  # CNN Max Pooling Filter Size Per Layer
            mp = [(5, 4), (1, 4), (1, 2)]

        if p is None:  # TODO: Correct? Output CNN Channels Per Layer
            p = [16, 32, 64]

        base_layers = []
        for i in range(len(mp)):
            base_layers.append(nn.Conv2d(in_channels, p[i], 3, padding=1))
            base_layers.append(nn.BatchNorm2d(p[i]))
            base_layers.append(nn.ReLU())
            base_layers.append(nn.MaxPool2d(kernel_size=mp[i], stride=mp[i]))
            in_channels = p[i]

        base_layers.append(nn.Flatten())
        self.base = nn.Sequential(*base_layers)

        self.gru = nn.GRU(input_size=128, hidden_size=gru_hidden_size, num_layers=2, bidirectional=True,
                          batch_first=True)  # TODO: What is hidden size?

        seld_layers = [nn.Linear(gru_hidden_size * 2, 128), nn.Linear(128, 3 * 3 * c), nn.Tanh()]
        self.seld = nn.Sequential(*seld_layers)

    def forward(self, x):
        base = self.base(x)
        gru, _ = self.gru(base)
        seld = self.seld(gru)
        b, t = seld.shape
        return seld.reshape((b, 3, 3, t // (3 * 3)))

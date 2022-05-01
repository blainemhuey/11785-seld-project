import torch
import torch.nn as nn

class Block(nn.Module):

    def forward(self, A):
        # import ipdb; ipdb.set_trace()
        Z = self.bn1(self.conv1.forward(A))
        Z = self.relu(Z)
        Z = self.bn2(self.conv2.forward(Z))
        if self.use_1x1:
            A = self.conv3(A)
            # still not done, have bn or not??
        Z += A
        return self.relu(Z)
    
    def forward_new(self, A):
        Z = A
        for layer in self.layers:
            Z = layer.forward(Z)
        if self.use_1x1:
            A = self.conv3(A)
        Z += A
        return self.relu(Z)


class (nn.Module):
    """
    ResBlock as described in https://arxiv.org/abs/1707.02921
    """

    def __init__(self, in_channels, activation=nn.ReLU, kernel_sizes=(3, 5)):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)

    def forward(self, x):
        Z = self.conv1.forward(A)
        Z = self.relu(Z)
        Z = self.conv2.forward(Z)
        Z += A
        return self.relu(Z)

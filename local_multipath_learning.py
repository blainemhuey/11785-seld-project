import torch
import torch.nn as nn


class MultiScaleResidualBlock(nn.Module):
    """
    Implementation of Multi-Scale Residual Block

    https://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf
    """

    def __init__(self, in_channels, activation=nn.ReLU, kernel_sizes=(3, 5)):
        super().__init__()
        kernel_s, kernel_p = kernel_sizes

        self.s1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_s, padding=kernel_s//2),
            activation()
        )

        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_p, padding=kernel_p//2),
            activation()
        )

        self.s2 = nn.Sequential(
            nn.Conv2d(2*in_channels, 2*in_channels, kernel_s, padding=kernel_s//2),
            activation()
        )

        self.p2 = nn.Sequential(
            nn.Conv2d(2*in_channels, 2*in_channels, kernel_p, padding=kernel_p//2),
            activation()
        )

        self.sp = nn.Conv2d(4*in_channels, in_channels, 1)

    def forward(self, x):
        s1 = self.s1(x)
        p1 = self.p1(x)
        s2 = self.s2(torch.concat((s1, p1), dim=1))
        p2 = self.p2(torch.concat((p1, s1), dim=1))
        sp = self.sp(torch.concat((s2, p2), dim=1))
        return sp + x

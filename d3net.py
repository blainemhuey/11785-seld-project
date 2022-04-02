import torch
import torch.nn as nn


class MultidilatedConv2dBlock(nn.Module):
    """
    Basic D3Net Conv2D extension that supports more than one dilation
    @see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True, padding_mode='zeros',
                 device=None, dilation_factors=tuple([1])):
        super().__init__()

        self.convolutions = []
        for d in dilation_factors:
            # Calculate the padding needed to have the same output shape
            padding = (d * (kernel_size - 1)) // 2

            # Make Parallel Convolution for Each Dilation Level
            new_conv = nn.Conv2d(in_channels, out_channels-in_channels, kernel_size, stride=stride,
                                 padding=padding, groups=groups, bias=bias, padding_mode=padding_mode,
                                 device=device, dilation=d)
            self.convolutions.append(new_conv)

    def forward(self, x):
        output = None
        for conv in self.convolutions:
            if output is None:
                output = conv(x)
            else:
                output = output + conv(x)
        return torch.concat((x, output), dim=1)


class D2Block(nn.Module):
    """
    D2 Block Implementation
    """

    def __init__(self, in_channels, kernel_size, growth_rate, bottleneck_channels, compression_rate, dilation_factors=tuple([1, 2, 4])):
        super().__init__()

        self.dilation_factors = dilation_factors
        self.growth_rate = growth_rate
        self.compression_rate = compression_rate

        # Add Bottleneck layer when in_channels > 4*k
        if in_channels > bottleneck_channels:
            self.bottleneck = nn.Conv2d(in_channels, bottleneck_channels, 1)
            in_channels = bottleneck_channels
        else:
            self.bottleneck = None

        # Create multidilated layers, with each having one more dilation until all dilations are used
        layers = []
        for i in range(len(dilation_factors)):
            selected_dilation_factors = dilation_factors[:i+1]
            layers.append(MultidilatedConv2dBlock(in_channels, in_channels + growth_rate, kernel_size,
                                                  dilation_factors=selected_dilation_factors))
            in_channels += growth_rate
        self.dilated_layers = layers

        # Add Compression layer
        # TODO: Compress layer not needed for audio source separation?  D3Net Paper says so
        self.compress = None  # nn.Conv2d()

    def forward(self, x):
        values = self.bottleneck(x) if self.bottleneck else x

        for i, l in enumerate(self.dilated_layers):
            values = l(values)

        return self.compress(values) if self.compress else values


class D3Block(nn.Module):
    """
    As per the D3Net Paper, the parameters characterizing a D3 block are:
    M - The number of D2 blocks
    L - The number of layers in each D2 block
    k - Growth rate
    B - Bottleneck layer channels
    c - Compression ratio (ignored here since it is only used for segmentation, not source separation)
    """

    def __init__(self, in_channels, M, L, k, B=None, c=None):
        super().__init__()

        kernel_size = 3  # TODO: Is this the correct size?
        if B is None:
            B = 4 * k  # Bottleneck size default specified in paper

        d2_blocks = []
        for block in range(M):
            d2_block = D2Block(in_channels, kernel_size, k, B, c, dilation_factors=[2**i for i in range(L)])
            if in_channels > B:
                in_channels = B + (k * L)
            else:
                in_channels += k * L
            d2_blocks.append(d2_block)
        self.d2_blocks = d2_blocks

    def forward(self, x):
        values = x
        for i, l in enumerate(self.d2_blocks):
            new_values = l(values)
            values = new_values
            # torch.concat((values, new_values), dim=1) # Test
        return values


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.layers = [
            D3Block(7, 4, 4, 16, 2),
            nn.Conv2d(66, 66//2, 1),  #stride?
            nn.AvgPool2d(2),
            D3Block(66//2, 4, 4, 24, 2),
            nn.Conv2d(98, 98//2, 1),
            nn.AvgPool2d(2),
            D3Block(98//2, 4, 4, 32, 2),
            nn.Conv2d(130, 130//2, 1),
            nn.AvgPool2d(2),
            D3Block(130//2, 4, 4, 40, 2),
            nn.AvgPool2d(2),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.GRU(32, 160, batch_first=True), #check for num_layers if hidden = 160 doesnt work!
        ]

        self.class_layer = nn.Linear(160*162, 117)

    def forward(self, A):
        for i, layer in enumerate(self.layers):
            A = layer(A)

        batch_size, size_a, size_b = A[0].shape
        A = self.class_layer(A[0].reshape((batch_size, size_a*size_b)))

        # x = self.laysers(A).reshape((3,3,13))
        return A.reshape((batch_size,3,3,13))

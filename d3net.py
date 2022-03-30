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
            new_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                 padding=padding, groups=groups, bias=bias, padding_mode=padding_mode,
                                 device=device, dilation=d)
            self.convolutions.append(new_conv)

    def forward(self, *args):
        output = None
        for conv, x in zip(self.convolutions, args):
            if output is None:
                output = conv(args)
            else:
                output = output + conv(
                    args)  # TODO: Different Dilations combined before or after module? Concatenate or add?
        return output


class D2Block(nn.Module):
    """
    D2 Block Implementation
    """

    def __init__(self, in_channels, kernel_size, growth_rate, bottleneck_channels, compression_rate, dilation_factors=tuple([1, 2, 4])):
        super().__init__()

        self.dilation_factors = dilation_factors
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
            selected_dilation_factors = dilation_factors[:i]
            layers.append(MultidilatedConv2dBlock(in_channels, in_channels * growth_rate, kernel_size,
                                                  dilation_factors=selected_dilation_factors))
            in_channels *= growth_rate
        self.dilated_layers = layers

        # Add Compression layer
        # TODO: Compress layer not needed for audio source separation?  D3Net Paper says so
        # self.compress = nn.Conv2d()

    def forward(self, x):
        values = [x if self.bottleneck is None else self.bottleneck(x)]

        for i, l in enumerate(self.layers):
            output = l(*values)
            values.append(output)

        return values[-1]


class D3Block(nn.Module):
    """
    As per the D3Net Paper, the parameters characterizing a D3 block are:
    M - The number of D2 blocks
    L - The number of layers in each D2 block
    k - Growth rate
    B - Bottleneck layer channels
    c - Compression ratio (ignored here since it is only used for segmentation, not source separation)
    """

    def __init__(self, M, L, k, B=None, c=None):
        super().__init__()

        kernel_size = 3  # TODO: Is this the correct size?
        if B is None:
            B = 4 * k  # Bottleneck size default specified in paper

        d2_blocks = []
        in_channels = 1
        for block in range(M):
            d2_block = D2Block(in_channels, kernel_size, k, B, c, dilation_factors=[2**i for i in range(L)])
            in_channels = B * (k ** L)
            d2_blocks.append(d2_block)
        self.d2_blocks = d2_blocks

    def forward(self, x):
        values = [x]
        for i, l in enumerate(self.d2_blocks):
            pass  # TODO: How to reconcile different output sizes?

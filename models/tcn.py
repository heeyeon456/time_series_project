import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import random
from collections import OrderedDict
import math


class Chomp1d(nn.Module):
    def __init__(self, size):
        super(Chomp1d, self).__init__()
        self.size = size

    def forward(self, x):
        return x[:, :, :-self.size].contiguous()


class TemporalBlock(nn.Module):
    """
    One block for dilated temporal convolution.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 kernel_size : int = 4,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 dropout: float = 0.1):
        """ Constructor

        Args:
            input_dim (int): Input Dimension
            hidden_size (int): Hidden layer dimension.
                               All layers have same dimension
            output_dim (int): Output Dimension
            kernel_size (int, optional): Kernel size for 1d Conv. Defaults to 4.
            stride (int, optional): Stride for 1d Conv. Defaults to 1.
            padding (int, optional): Padding size for 1d Conv. Defaults to 0.
            dilation (int, optional): Dilation size for Dilated Convolution. Defaults to 1. (No dilation)
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(nn.Conv1d(in_channels=input_dim,
                                           out_channels=output_dim,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(size=padding)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = weight_norm(nn.Conv1d(in_channels=input_dim,
                                           out_channels=output_dim,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(size=kernel_size-1)
        #self.network = nn.Sequential(self.conv1, self.chomp1, self.relu, self.dropout,
        #                             self.conv2, self.chomp2, self.relu, self.dropout)

        self.network = nn.Sequential(self.conv1, self.chomp1, self.relu, self.dropout)
        self.downsample = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else None


        print(self.network)
        self._init_weights()

    def _init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        h, w, c = x.size()
        res = x if self.downsample is None else self.downsample(x)

        out = self.network(x)
        # residual connection
        out = self.relu(out + res)
        return out


class TemporalCNNNetwork(nn.Module):
    """
    Temporal Convolution Network
    reference code : https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
    """
    def __init__(self,
                 input_dim: int,
                 num_channels: list,
                 kernel_size: int = 2,
                 dropout : float = 0.2):
        """ Constructor

        Args:
            input_dim (int): Input Dimension
            num_channel (list): Output Dimension list for each temporal block.
            kernel_size (int, optional): Kernel size for 1d Conv. Defaults to 4.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """

        super(TemporalCNNNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        h, w, c = out.size()
        out = out.reshape((h, w*c))
        return out


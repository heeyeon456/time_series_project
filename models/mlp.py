import torch
import torch.nn as nn

import random
from collections import OrderedDict

class MLPBlock(nn.Module):
    """
    Base MLP Block for MLPNetwork
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 layer_num: int):

        super(MLPBlock, self).__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

        self.network = nn.Sequential(self.flatten, self.fc, self.relu)

        if layer_num == 0:
            self.network = nn.Sequential(self.flatten, self.fc, self.relu)
        else:
            self.network = nn.Sequential(self.fc, self.relu)

        print(self.network)
        self.network.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
        return m

    def forward(self, x):
        return self.network(x)


class MLPNetwork(nn.Module):
    """
    Base MLP network structure.
    It consists of 3 FC layers.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_size: int,
                 output_dim: int,
                 num_layers: int):
        """[summary]

        Args:
            input_dim (int): Input Dimension
            hidden_size (int): Hidden layer dimension.
                               All layers have same dimension
            output_dim (int): Output Dimension
            num_layers (int) : Number of MLP layers
        """
        super(MLPNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        layers = []

        for i in range(num_layers):
            input_size = input_dim if i == 0 else hidden_size
            output_size = output_dim if i == num_layers-1 else hidden_size

            layers += [MLPBlock(input_size, output_size, i)]

        self.network = nn.Sequential(*layers)
        self.network.apply(self._init_weights)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)
        return m

    def forward(self, x):
        return self.network(x)


class RegressorNetwork(nn.Module):
    """
    Network for merging previous target network and additional feature network.
    It consists of one fc layer.
    """
    def __init__(self, input1_dim, input2_dim, output_dim):
        """[summary]

        Args:
            input1_dim ([type]): Output dimension for first network (prev target network)
            input2_dim ([type]): Output dimension for second network (additional feat network)
            output_dim ([type]): prediction output dimension
        """
        super(RegressorNetwork, self).__init__()
        self.input_dim = input1_dim + input2_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, self.output_dim)
        self.relu = nn.ReLU()

    def forward(self, input1, input2):
        out = torch.cat((input1, input2), dim=1)
        out = self.fc1(out)
        out = self.relu(out)
        return out

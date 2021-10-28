import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMNetwork(nn.Module):
    def __init__(self,
               input_dim: int,
               hidden_size: int,
               output_dim: int,
               num_feat: int,
               device: str,
               layer_dim: int = 1,
               dropout:int = 0.1):
        """ LSTM Network

        Args:
            input_dim (int): Number of input dimension for each sequential element
            hidden_size (int): Hidden layer size for each lstm cell
            output_dim (int): Output dimension of FC layer
            num_feat (int) : Number of time-series input features
            device (str): Which device to use (cpu or gpu)
            layer_dim (int, optional): Number of LSTM Layer. Defaults to 1.
            dropout (int, optional): Dropout probability. Defaults to 0.1.
        """
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = layer_dim

        self.input_dim = num_feat * input_dim

        self.device = device

        #Building the LSTM
        self.network = nn.LSTM(input_size=self.input_dim,
                               hidden_size=hidden_size,
                               num_layers=layer_dim,
                               dropout=0.1,
                               batch_first=True)
        print(self.network)
        self._init_weights(self.network)

        # Readout layer
        self.fc = nn.Linear(hidden_size, output_dim)

    def _init_weights(self, net):
        for layer_p in net._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    #print(p, net.__getattr__(p))
                    nn.init.normal(net.__getattr__(p), 0.0, 0.02)

    def forward(self, x):
        # Initializing the hidden state with zeros
        # (input, hx, batch_sizes)
        h, w, c = x.size()
        x = x.view(-1, (w*c) // self.input_dim, self.input_dim)

        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)


        #One time step (the last one perhaps?)
        out, (hn,cn) = self.network(x, (h0, c0))
        #print("out:", out.size())

        # Indexing hidden state of the last time step
        # out.size() --> ??
        #out[:,-1,:] --> is it going to be 100,100
        out = self.fc(out[:,-1,:])
        # out.size() --> 100,1
        return out
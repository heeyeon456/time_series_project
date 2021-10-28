import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.autograd import Variable

class CNNLSTMNetwork(nn.Module):
    def __init__(self,
               input_dim: int,
               hidden_size: int,
               output_dim: int,
               num_feat: int,
               device: str,
               layer_dim: int = 1,
               dropout:int = 0.0):
        """ CNN_LSTM Network
        reference code : https://github.com/pranoyr/cnn-lstm/blob/master/models/cnnlstm.py

        Args:
            input_dim (int): Number of input dimension for each sequential element
            hidden_size (int): Hidden layer size for each lstm cell
            output_dim (int): Output dimension of FC layer
            num_feat (int) : Number of time-series input features
            device (str): Which device to use (cpu or gpu)
            layer_dim (int, optional): Number of LSTM Layer. Defaults to 1.
            dropout (int, optional): Dropout probability. Defaults to 0.1.
        """
        super(CNNLSTMNetwork, self).__init__()
        self.hidden_dim = hidden_size
        self.layer_dim = layer_dim
        self.input_dim = input_dim

        self.device = device
        self.conv1 = weight_norm(
            nn.Conv1d(in_channels=self.input_dim,
                      out_channels=self.hidden_dim,
                      kernel_size=num_feat))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm((self.hidden_dim, 1))

        #Building the LSTM
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=layer_dim,
                            dropout=dropout,
                            batch_first=True)
        self._init_weights(self.lstm)

        # Readout layer
        self.fc = nn.Linear(hidden_size, output_dim)

    def _init_weights(self, net):
        for layer_p in net._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.normal(net.__getattr__(p), 0.0, 0.02)

    def __repr__(self):
        model_info = ""
        model_list = [self.conv1, self.relu, self.dropout, self.ln, self.lstm]
        for i in range(5):
            model_info += "({}): {}\n".format(i, model_list[i])

        return model_info


    def forward(self, x):
        h, w, c = x.size()
        feat_len = w // self.input_dim
        x = x.view(-1, feat_len, self.input_dim, c)

        # Initializing the hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)).to(self.device)

        for t in range(feat_len):
            x_tmp = x[:, t, :, :]
            x_tmp = self.conv1(x_tmp)
            x_tmp = self.relu(x_tmp)
            x_tmp = self.dropout(x_tmp)
            x_tmp = self.ln(x_tmp)

            h, w, c = x_tmp.size()
            x_tmp = x_tmp.reshape((h, 1, w*c))
            out, (h0, c0) = self.lstm(x_tmp, (h0, c0))

        out = self.fc(out[:,-1,:])
        return out
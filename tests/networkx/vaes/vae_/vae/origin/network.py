import torch
import torch.nn as nn
from ..base import NetworkBase
from layertools import set_linear
from typing import *

class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def encode(self, x):
        h_e = self.encoder_net(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)

        return mu_e, log_var_e

    def forward(self, x):
        return self.encode(x)

class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution='categorical', num_vals=None):
        super(Decoder, self).__init__()

        self.decoder_net = decoder_net
        self.distribution = distribution
        self.num_vals = num_vals

    def decode(self, z):
        h_d = self.decoder_net(z)
        if self.distribution == 'categorical':
            b = h_d.shape[0]
            d = h_d.shape[1] // self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)
            return [mu_d]
        elif self.distribution == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
            return [mu_d]

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

    def forward(self, z):
        return self.decode(z)

class Network(NetworkBase):
    data_demand = None

    def __init__(self,
                 input_size: Tuple,
                 latent_dim:int=10,
                 dim_channel_list:List=[20,30,50],
                 encoder_net=None, decoder_net=None,
                 distribution='categorical',
                 num_vals= 10,):
        super(Network, self).__init__()
        if encoder_net is None:
            encoder_net = set_linear(input_size, (2 * latent_dim,),dim_channel_list)
        if decoder_net is None:
            decoder_net = set_linear((latent_dim,), (num_vals * input_size[-1],), dim_channel_list)
        self.num_vals = num_vals
        self.latent_dim = latent_dim
        self.D = input_size[-1]
        self.encoder = Encoder(encoder_net)
        self.decoder = Decoder(decoder_net, distribution, num_vals)




from vae.dele.vaebase import VAENetworkBase
import torch.nn as nn
import torch
from layertools import log_normal_diag,log_categorical,log_bernoulli,set_linear
from typing import *

class Encoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net
    #重参数化
    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        h_e = self.encoder_net(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)

    def sample(self, x=None, mu_e=None, log_var_e=None):
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log-scale can`t be None!')
        z = self.reparameterization(mu_e, log_var_e)
        return z

    # 极大似然
    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-scale and z can`t be None!')

        return log_normal_diag(z, mu_e, log_var_e)

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

    def sample(self, z):
        outs = self.decode(z)

        if self.distribution == 'categorical':
            mu_d = outs[0]
            b = mu_d.shape[0]
            m = mu_d.shape[1]
            mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)
            p = mu_d.view(-1, self.num_vals)
            x_new = torch.multinomial(p, num_samples=1).view(b, m)

        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            x_new = torch.bernoulli(mu_d)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return x_new

    def log_prob(self, x, z):
        outs = self.decode(z)
        if self.distribution == 'categorical':
            mu_d = outs[0]
            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')
        return log_p


    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(x)


class Network(VAENetworkBase):
    data_demand = None

    def __init__(self,
                 input_size: Tuple,
                 latent_dim:int,
                 dim_channel_list:List,
                 encoder_net=None, decoder_net=None,
                 scatter_dim = 10,
                 distribution='categorical',
                 num_vals=None):

        if encoder_net is None:
            encoder_net = set_linear(input_size, 2 * latent_dim,dim_channel_list)
        if decoder_net is None:
            decoder_net = set_linear(latent_dim, scatter_dim * input_size[-1], dim_channel_list)

        self.encoder = Encoder(encoder_net)
        self.decoder = Decoder(decoder_net, distribution, num_vals)
        super(Network, self).__init__(input_size=input_size,
                         encoder_network = self.encoder,
                         decoder_network = self.decoder,
                         latent_dim=16)
        self.distribution = distribution
        self.num_vals = num_vals
        self.dim_channel_list = dim_channel_list

    def forward(self, batch):
        ...

    def get_information(self):
        return super().get_information() + f"""scatter_dim : {self.scatter_dim},
Latent_dim : {self.Latent_dim}
distribution : {self.distribution}
dim_channel_list : {self.dim_channel_list}
num_vals : {self.num_vals}
"""

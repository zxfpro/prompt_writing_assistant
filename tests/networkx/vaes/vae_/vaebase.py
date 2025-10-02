



from .nn_function import set_linear
from .base import ModelBase,NetworkBase
from .utils import log_normal_diag,log_categorical,log_bernoulli
from typing import *
import torch.nn as nn
import torch

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

class VAENetworkBase(NetworkBase):
    data_demand = None

    def __init__(self,input_size:Tuple,
                 nnseq:Callable[[torch.Tensor],object] or None=None,
                 dim_channel_list=[20, 40, 60],
                 scatter_dim=64,
                 Latent_dim = 16,):

        if nnseq is None:
            encoder_net = set_linear(input_size, 2 * Latent_dim, dim_channel_list)
            decoder_net = set_linear(Latent_dim, scatter_dim * input_size[-1], dim_channel_list)
            network_obj = {'encoder':encoder_net,
                           'decoder':decoder_net}


        else:
            network_obj = nnseq

        super().__init__(input_size=input_size,dim_channel_list=dim_channel_list,
                         network=network_obj)


        self.Latent_dim = Latent_dim
        self.scatter_dim = scatter_dim

    def forward(self, batch):
        return self.network(batch)

    def get_information(self):
        return super().get_information() + f"""scatter_dim : {self.scatter_dim},
Latent_dim : {self.Latent_dim}
"""

class VAEModelBase(ModelBase):
    name =  'VAEModelBase'
    def __init__(self, network:VAENetworkBase,lr,device='cpu'):
        self.encoder = network['encoder']
        self.decoder = network['decoder']
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        super().__init__(optimizer=optimizer,lr=lr,device=device)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def forward(self, batch_data,cond=None):
        mu_e, log_var_e = self.encoder(batch_data)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        RE = self.decoder.log_prob(batch_data, z)
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)
        return -(RE + KL).mean()

    def train(self,batch_data,cond=None):
        self.optimizer.zero_grad()
        loss = self.forward(batch_data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def generate(self, batch_size, cond=None):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)

    def get_information(self):
        return super().get_information() +""""""


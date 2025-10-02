from gmodel.base import ModelBase,NetworkBase
import torch.nn as nn
import torch

import numpy as np
import torch
import torch.nn.functional as F

PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-5

def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    """
    categorical 方法

    num_classes 离散化的维度 或者是类别数量

    """
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_bernoulli(x, p, reduction=None, dim=None):
    """
    bernoulli 分布的计算

    """
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    # 正态分布
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_standard_normal(x, reduction=None, dim=None):
    #标准正态
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

# Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
#  Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743

def log_min_exp(a, b, epsilon=1e-8):
    """
    Source: https://github.com/jornpeters/integer_discrete_flows
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
    log(exp(a) - exp(b))
    c + log(exp(a-c) - exp(b-c))
    a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y

def log_integer_probability(x, mean, logscale):
    scale = torch.exp(logscale)

    logp = log_min_exp(
      F.logsigmoid((x + 0.5 - mean) / scale),
      F.logsigmoid((x - 0.5 - mean) / scale))

    return logp

def log_integer_probability_standard(x):
    logp = log_min_exp(
      F.logsigmoid(x + 0.5),
      F.logsigmoid(x - 0.5))

    return logp




class GTMPrior(nn.Module):
    def __init__(self, L, gtm_net, num_components, u_min=-1., u_max=1.):
        super(GTMPrior, self).__init__()

        self.L = L

        # 2D manifold
        self.u = torch.zeros(num_components**2, 2) # K**2 x 2
        u1 = torch.linspace(u_min, u_max, steps=num_components)
        u2 = torch.linspace(u_min, u_max, steps=num_components)

        k = 0
        for i in range(num_components):
            for j in range(num_components):
                self.u[k,0] = u1[i]
                self.u[k,1] = u2[j]
                k = k + 1

        # gtm network: u -> z
        self.gtm_net = gtm_net

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components**2, 1, 1))

    def get_params(self):
        # u->z
        h_gtm = self.gtm_net(self.u) #K**2 x 2L
        mean_gtm, logvar_gtm = torch.chunk(h_gtm, 2, dim=1) # K**2 x L and K**2 x L
        return mean_gtm, logvar_gtm

    def sample(self, batch_size):
        # u->z
        mean_gtm, logvar_gtm = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.L)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = mean_gtm[[indx]] + eps[[i]] * torch.exp(logvar_gtm[[indx]])
            else:
                z = torch.cat((z, mean_gtm[[indx]] + eps[[i]] * torch.exp(logvar_gtm[[indx]])), 0)
        return z

    def log_prob(self, z):
        # u->z
        mean_gtm, logvar_gtm = self.get_params()

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        mean_gtm = mean_gtm.unsqueeze(1) # K**2 x 1 x L
        logvar_gtm = logvar_gtm.unsqueeze(1) # K**2 x 1 x L

        w = F.softmax(self.w, dim=0)

        log_p = log_normal_diag(z, mean_gtm, logvar_gtm) + torch.log(w) # K**2 x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob


class VampPrior(nn.Module):
    def __init__(self, L, D, num_vals, encoder, num_components, data=None):
        super(VampPrior, self).__init__()

        self.L = L
        self.D = D
        self.num_vals = num_vals

        self.encoder = encoder

        # pseudoinputs
        u = torch.rand(num_components, D) * self.num_vals
        self.u = nn.Parameter(u)

        # mixing weights
        self.w = nn.Parameter(torch.zeros(self.u.shape[0], 1, 1)) # K x 1 x 1

    def get_params(self):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.encoder.encode(self.u) #(K x L), (K x L)
        return mean_vampprior, logvar_vampprior

    def sample(self, batch_size):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0) # K x 1 x 1
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.L)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = mean_vampprior[[indx]] + eps[[i]] * torch.exp(logvar_vampprior[[indx]])
            else:
                z = torch.cat((z, mean_vampprior[[indx]] + eps[[i]] * torch.exp(logvar_vampprior[[indx]])), 0)
        return z

    def log_prob(self, z):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params() # (K x L) & (K x L)

        # mixing probabilities
        w = F.softmax(self.w, dim=0) # K x 1 x 1

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        mean_vampprior = mean_vampprior.unsqueeze(1) # K x 1 x L
        logvar_vampprior = logvar_vampprior.unsqueeze(1) # K x 1 x L

        log_p = log_normal_diag(z, mean_vampprior, logvar_vampprior) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob # B

class GTMVampPrior(nn.Module):
    def __init__(self, L, D, gtm_net, encoder, num_points, u_min=-10., u_max=10., num_vals=255):
        super(GTMVampPrior, self).__init__()

        self.L = L
        self.D = D
        self.num_vals = num_vals

        self.encoder = encoder

        # 2D manifold
        self.u = torch.zeros(num_points**2, 2) # K**2 x 2
        u1 = torch.linspace(u_min, u_max, steps=num_points)
        u2 = torch.linspace(u_min, u_max, steps=num_points)

        k = 0
        for i in range(num_points):
            for j in range(num_points):
                self.u[k,0] = u1[i]
                self.u[k,1] = u2[j]
                k = k + 1

        # gtm network: u -> x
        self.gtm_net = gtm_net

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_points**2, 1, 1))

    def get_params(self):
        # u->gtm_net->u_x
        h_gtm = self.gtm_net(self.u) #K x D
        h_gtm = h_gtm * self.num_vals
        # u_x->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.encoder.encode(h_gtm) #(K x L), (K x L)
        return mean_vampprior, logvar_vampprior

    def sample(self, batch_size):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.L)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = mean_vampprior[[indx]] + eps[[i]] * torch.exp(logvar_vampprior[[indx]])
            else:
                z = torch.cat((z, mean_vampprior[[indx]] + eps[[i]] * torch.exp(logvar_vampprior[[indx]])), 0)
        return z

    def log_prob(self, z):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        mean_vampprior = mean_vampprior.unsqueeze(1) # K x 1 x L
        logvar_vampprior = logvar_vampprior.unsqueeze(1) # K x 1 x L

        log_p = log_normal_diag(z, mean_vampprior, logvar_vampprior) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob

class MoGPrior(nn.Module):
    def __init__(self, L, num_components,multiplier):
        super(MoGPrior, self).__init__()

        self.L = L
        self.num_components = num_components

        # params
        self.means = nn.Parameter(torch.randn(num_components, self.L)*multiplier)
        self.logvars = nn.Parameter(torch.randn(num_components, self.L))

        # mixing weights
        self.w = nn.Parameter(torch.zeros(num_components, 1, 1))

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)
        w = w.squeeze()

        # pick components
        indexes = torch.multinomial(w, batch_size, replacement=True)

        # means and logvars
        eps = torch.randn(batch_size, self.L)
        for i in range(batch_size):
            indx = indexes[i]
            if i == 0:
                z = means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])
            else:
                z = torch.cat((z, means[[indx]] + eps[[i]] * torch.exp(logvars[[indx]])), 0)
        return z

    def log_prob(self, z):
        # mu, lof_var
        means, logvars = self.get_params()

        # mixing probabilities
        w = F.softmax(self.w, dim=0)

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        means = means.unsqueeze(1) # K x 1 x L
        logvars = logvars.unsqueeze(1) # K x 1 x L

        log_p = log_normal_diag(z, means, logvars) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob


class StandardPrior(nn.Module):
    def __init__(self, L=2):
        super(StandardPrior, self).__init__()

        self.L = L

        # params weights
        self.means = torch.zeros(1, L)
        self.logvars = torch.zeros(1, L)

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        return torch.randn(batch_size, self.L)

    def log_prob(self, z):
        return log_standard_normal(z)



class Prior(nn.Module):
    def __init__(self, L):
        super(Prior, self).__init__()
        self.L = L

    def sample(self, batch_size):
        z = torch.randn((batch_size, self.L))
        return z

    def log_prob(self, z):
        return log_standard_normal(z)




class FlowPrior(nn.Module):
    def __init__(self, nets, nett, num_flows, D=2):
        super(FlowPrior, self).__init__()

        self.D = D

        self.t = torch.nn.ModuleList([nett() for _ in range(num_flows)])
        self.s = torch.nn.ModuleList([nets() for _ in range(num_flows)])
        self.num_flows = num_flows

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)

        s = self.s[index](xa)
        t = self.t[index](xa)

        if forward:
            #yb = f^{-1}(x)
            yb = (xb - t) * torch.exp(-s)
        else:
            #xb = f(y)
            yb = torch.exp(s) * xb + t

        return torch.cat((xa, yb), 1), s

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s = self.coupling(z, i, forward=True)
            z = self.permute(z)
            log_det_J = log_det_J - s.sum(dim=1)

        return z, log_det_J

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x, _ = self.coupling(x, i, forward=False)

        return x

    def sample(self, batch_size):
        z = torch.randn(batch_size, self.D)
        x = self.f_inv(z)
        return x.view(-1, self.D)

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        log_p = (log_standard_normal(z) + log_det_J.unsqueeze(1))
        return log_p


# network
class Encoder(nn.Module):
    def __init__(self,encoder):
        super().__init__()
        self.encoder = encoder

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(self, x):
        h_e = self.encoder(x)
        mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
        return mu_e, log_var_e

    def sample(self, x=None, mu_e=None, log_var_e=None):
        if (mu_e is None) and (log_var_e is None):
            mu_e, log_var_e = self.encode(x)
        else:
            if (mu_e is None) or (log_var_e is None):
                raise ValueError('mu and log-scale can`t be None!')
        z = self.reparameterization(mu_e, log_var_e)
        return z

    def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
        if x is not None:
            mu_e, log_var_e = self.encode(x)
            z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
        else:
            if (mu_e is None) or (log_var_e is None) or (z is None):
                raise ValueError('mu, log-scale and z can`t be None!')

        return log_normal_diag(z, mu_e, log_var_e)

    def forward(self, x, type='log_prob'):
        assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x)
        else:
            return self.sample(x)
class Decoder(nn.Module):
    def __init__(self, decoder_net, distribution='categorical', num_vals=None):
        super(Decoder, self).__init__()

        self.decoder = decoder_net
        self.distribution = distribution
        self.num_vals=num_vals

    def decode(self, z):
        h_d = self.decoder(z)
        if self.distribution == 'categorical':
            b = h_d.shape[0]
            d = h_d.shape[1]//self.num_vals
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


class VAENetwork(NetworkBase):
    data_demand = None
    def __init__(self,input_size,L=16,num_vals=64):
        """
        input_size  : window_size D
        L 隐空间的维度   原始为16    更新为2
        num——vals  ： digit——dim  分类维数
        """
        super(VAENetwork, self).__init__()
        self.input_size = input_size
        self.encoder_net,self.decoder_net = self.set_linear(window_size=input_size[-1],L=L,num_vals=num_vals)
        self.num_vals = num_vals
        self.L = L

    def set_linear(self,window_size,num_vals,L = 2,M = 256):
        """
        D = window_size
        """

        encoder_net = nn.Sequential(nn.Linear(window_size, M), nn.LeakyReLU(),
                                    nn.Linear(M, M), nn.LeakyReLU(),
                                    nn.Linear(M, 2 * L))

        decoder_net = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),
                                    nn.Linear(M, M), nn.LeakyReLU(),
                                    nn.Linear(M, num_vals * window_size))
        return encoder_net,decoder_net
    
    def forward(self,batch):
        pass

    def __repr__(self):
        return f"""
data_demand     :{self.data_demand}
input_size      :{self.input_size}
num_vals       :{self.num_vals}
L              :{self.L}
"""


class VAEModel(ModelBase):
    name = 'VAEpro'
    def __init__(self, network:VAENetwork,lr,digit_dim=64,prior_name='vampprior', z_dim=16, likelihood_type='categorical',device='cpu'):
        super(VAEModel, self).__init__()
        encoder = network.encoder_net
        decoder = network.decoder_net
        self.encoder = Encoder(encoder)
        self.decoder = Decoder(decoder,distribution = likelihood_type,num_vals=digit_dim)

        if prior_name in ['standard', 'flow2']:
            num_components = 1
        elif prior_name[0:3] == 'gtm':
            num_components = 4
        else:
            num_components = 4 ** 2

        self.L = network.L
        self.D = network.input_size[-1]
        self.num_vals = network.num_vals
        self.prior_name = prior_name
        # Second, we initialize the prior

        if prior_name  == 'origin':
            self.prior = Prior(L=z_dim)
        elif prior_name == 'vampprior':
            self.prior = VampPrior(L=self.L, D=self.D, num_vals=self.num_vals, encoder=self.encoder, num_components=num_components)
        elif prior_name == 'standard':
            self.prior = StandardPrior(L=self.L)
        elif prior_name == 'gtm':

            gtm_net = nn.Sequential(nn.Linear(2, 256), nn.Tanh(),
                                    nn.Linear(256, 256), nn.Tanh(),
                                    nn.Linear(256, 2 * self.L))

            self.prior = GTMPrior(L=self.L, gtm_net=gtm_net, num_components=num_components, u_min=-10., u_max=10.)
        elif prior_name == 'gtm-vampprior':
            gtm_net_vamp = nn.Sequential(nn.Linear(2, 256), nn.Tanh(),
                                         nn.Linear(256, 256), nn.Tanh(),
                                         nn.Linear(256, self.D), nn.Sigmoid())
            self.prior = GTMVampPrior(L=self.L, D=self.D, gtm_net=gtm_net_vamp, encoder=encoder, num_points=num_components,
                                 u_min=-10., u_max=10., num_vals=self.num_vals)
        elif prior_name == 'flow2':
            num_flows = 3

            # scale (s) network
            nets = lambda: nn.Sequential(nn.Linear(self.L // 2, 256), nn.LeakyReLU(),
                                         nn.Linear(256, 256), nn.LeakyReLU(),
                                         nn.Linear(256, self.L // 2), nn.Tanh())

            # translation (t) network
            nett = lambda: nn.Sequential(nn.Linear(self.L // 2, 256), nn.LeakyReLU(),
                                         nn.Linear(256, 256), nn.LeakyReLU(),
                                         nn.Linear(256, self.L // 2))

            self.prior = FlowPrior(nets, nett, num_flows=num_flows, D=self.L)
        self.digit_dim = digit_dim
        self.optimizer = torch.optim.Adamax([p for p in self.parameters() if p.requires_grad == True], lr=lr)
        self.lr = lr
        self.device = device


    def forward(self,batch_data,reduction='avg'):
        X,Y = batch_data
        X = X.type(torch.FloatTensor)
        mu_e, log_var_e = self.encoder.encode(X)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)

        RE = self.decoder.log_prob(X, z)
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)

        error = 0
        if np.isnan(RE.detach().numpy()).any():
            print('RE {}'.format(RE))
            error = 1
        if np.isnan(KL.detach().numpy()).any():
            print('RE {}'.format(KL))
            error = 1

        if error == 1:
            raise ValueError()

        if reduction == 'sum':
            return -(RE + KL).sum()
        else:
            return -(RE + KL).mean()


    def train(self, batch_data):
        loss = self.forward(batch_data)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def generate(self, batch_size, cond=None):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)

    def __repr__(self):
        return f"""
name            :{self.name}
prior_name      :{self.prior_name}
digit_dim       :{self.digit_dim}
lr              :{self.lr}
device          :{self.device}
        """


import torch.nn as nn
from gmodel.base import ModelBase,NetworkBase

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


    # def set_linear(self,input_size,dim_channel_list):
    #     in_channel = tuple_flatten(input_size)
    #     self.in_channel = in_channel
    #     network = nn.Sequential(Flatten(),
    #             MLP([in_channel,*dim_channel_list,self.scatter_dim],shape_len=self.shape_len,active='elu'))
    #     return network

class HierarchicalVAEnetwork(NetworkBase):
    data_demand = None
    def __init__(self,input_size,L = 16,num_vals=17,dim_channel_list=[56,128,64]):
        super(HierarchicalVAEnetwork, self).__init__()
        self.input_size = input_size
        D = input_size[-1]
        M = 256  # the number of neurons in scale (s) and translation (t) nets



        self.nn_r_1 = nn.Sequential(nn.Linear(D, M), GELU(),
                                   nn.BatchNorm1d(M),
                               nn.Linear(M, M), nn.GELU()
                              )

        self.nn_r_2 = nn.Sequential(nn.Linear(M, M), GELU(),
                                   nn.BatchNorm1d(M),
                               nn.Linear(M, M), nn.LeakyReLU()
                              )

        self.nn_delta_1 = nn.Sequential(nn.Linear(M, M), GELU(),
                                   nn.BatchNorm1d(M),
                                   nn.Linear(M, 2 * (L * 2)),
                                   )

        self.nn_delta_2 = nn.Sequential(nn.Linear(M, M), GELU(),
                                   nn.BatchNorm1d(M),
                                   nn.Linear(M, 2 * L),
                                   )

        self.nn_z_1 = nn.Sequential(nn.Linear(L, M), GELU(),
                               nn.BatchNorm1d(M),
                               nn.Linear(M, 2 * (L * 2))
                              )

        self.nn_x = nn.Sequential(nn.Linear(L * 2, M), GELU(),
                             nn.BatchNorm1d(M),
                             nn.Linear(M,M), GELU(),
                             nn.BatchNorm1d(M),
                             nn.Linear(M, D * num_vals)
                            )
    def forward(self,batch):
        pass


class HierarchicalVAEmodel(ModelBase):
    name = 'HVAE'
    def __init__(self, network,lr,digit_dim=17, D=30, L=16,
                 likelihood_type='categorical',device='cpu'):
        super(HierarchicalVAEmodel, self).__init__()
        # D window
        # nn_r_1, nn_r_2, nn_delta_1, nn_delta_2, nn_z_1, nn_x,
        # bottom-up path
        self.nn_r_1 = network.nn_r_1.to(device)
        self.nn_r_2 = network.nn_r_2.to(device)

        self.nn_delta_1 = network.nn_delta_1.to(device)
        self.nn_delta_2 = network.nn_delta_2.to(device)

        # top-down path
        self.nn_z_1 = network.nn_z_1.to(device)
        self.nn_x = network.nn_x.to(device)

        # other params
        self.D = D
        self.L = L

        self.num_vals = digit_dim

        self.likelihood_type = likelihood_type
        self.optimizer = torch.optim.Adamax([p for p in self.parameters() if p.requires_grad == True], lr=lr)
        self.lr = lr
        self.device = device


    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x,cond, reduction='avg'):
        # =====
        # bottom-up
        # step 1
        r_1 = self.nn_r_1(x)
        r_2 = self.nn_r_2(r_1)

        # step 2
        delta_1 = self.nn_delta_1(r_1)
        delta_mu_1, delta_log_var_1 = torch.chunk(delta_1, 2, dim=1)
        delta_log_var_1 = F.hardtanh(delta_log_var_1, -7., 2.)

        # step 3
        delta_2 = self.nn_delta_2(r_2)
        delta_mu_2, delta_log_var_2 = torch.chunk(delta_2, 2, dim=1)
        delta_log_var_2 = F.hardtanh(delta_log_var_2, -7., 2.)

        # top-down
        # step 4
        z_2 = self.reparameterization(delta_mu_2, delta_log_var_2)

        # step 5
        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)

        # step 6
        z_1 = self.reparameterization(mu_1 + delta_mu_1, log_var_1 + delta_log_var_1)

        # step 7
        h_d = self.nn_x(z_1)

        if self.likelihood_type == 'categorical':
            b = h_d.shape[0]
            d = h_d.shape[1] // self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)

        elif self.likelihood_type == 'bernoulli':
            mu_d = torch.sigmoid(h_d)

        # =====ELBO
        # RE
        if self.likelihood_type == 'categorical':
            RE = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.likelihood_type == 'bernoulli':
            RE = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        # KL
        KL_z_2 = 0.5 * (delta_mu_2 ** 2 + torch.exp(delta_log_var_2) - delta_log_var_2 - 1).sum(-1)
        KL_z_1 = 0.5 * (delta_mu_1 ** 2 / torch.exp(log_var_1) + torch.exp(delta_log_var_1) - \
                        delta_log_var_1 - 1).sum(-1)

        KL = KL_z_1 + KL_z_2

        error = 0
        if np.isnan(RE.cpu().detach().numpy()).any():
            print('RE {}'.format(RE))
            print('KL {}'.format(KL))
            error = 1
        if np.isnan(KL.cpu().detach().numpy()).any():
            print('RE {}'.format(RE))
            print('KL {}'.format(KL))
            error = 1

        if error == 1:
            raise ValueError()

        # Final ELBO
        if reduction == 'sum':
            loss = -(RE - KL).sum()
        else:
            loss = -(RE - KL).mean()

        return loss

    def generate(self, batch_size=64,cond=None):
        # step 1
        z_2 = torch.randn(batch_size, self.L).to(self.device)
        # step 2
        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        # step 3
        z_1 = self.reparameterization(mu_1, log_var_1)

        # step 4
        h_d = self.nn_x(z_1)

        if self.likelihood_type == 'categorical':
            b = batch_size
            d = h_d.shape[1] // self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)
            # step 5
            p = mu_d.view(-1, self.num_vals)
            x_new = torch.multinomial(p, num_samples=1).view(b, d)
        elif self.likelihood_type == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
            # step 5
            x_new = torch.bernoulli(mu_d)
        return x_new

    def train(self, batch_data):
        loss = self.forward(*batch_data)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()





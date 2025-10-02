from layertools import log_normal_diag,log_categorical,log_bernoulli
import torch



# 假设隐空间分布
class Prior():
    def __init__(self,Latent):
        """
        :param Latent: 隐空间维度
        默认为正态分布
        """
        self.Latent = Latent
    def sample(self, batch_size):
        """
        表示从分布中采样
        """
        z = torch.randn((batch_size, self.Latent))
        return z
    def log_prob(self, z):
        """
        表示Ln(p(x))
        """
        zero = torch.Tensor([0])
        return log_normal_diag(z,zero,zero)
        #mu 为均值 lagvar 为log方差 所以 0，0 为均值为0 方差为1的正态分布


import torch.nn as nn
import torch.nn.functional as F
class GTMPrior():
    def __init__(self, L, gtm_net, num_components, u_min=-1., u_max=1.):

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

# 编码器分布q(z|x)
class Encoder():
    def __init__(self):
        pass

    def sample(self, mu, log_var):
        """
        得到隐空间的均值和 ln方差
        从p(z|x)中采样
        """
        z = self.reparameterization(mu, log_var)
        return z
    def log_prob(self, z, mu_e, log_var_e):
        """
        得到Ln(q(z|x))
        """
        return log_normal_diag(z, mu_e, log_var_e)

    def reparameterization(self, mu, log_var):
        # 重参数化
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

class Decoder():
    def __init__(self,num_vals ,distribution='categorical'):
        self.distribution = distribution
        self.num_vals = num_vals

    def sample(self, devode_model,z):
        outs = devode_model(z)

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

    def log_prob(self, x, outs):
        if self.distribution == 'categorical':
            mu_d = outs[0]
            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')
        return log_p

class Process():
    """
    ().
    """
    def __init__(self,latent,D,num_vals= None,distribution = 'categorical',gtm_type='gtm',num_components = 4):
        if gtm_type == 'gtm':
            gtm_net = nn.Sequential(nn.Linear(2, 256), nn.Tanh(),
                                    nn.Linear(256, 256), nn.Tanh(),
                                    nn.Linear(256, 2 * latent))
            self.prior = GTMPrior(L=latent, gtm_net=gtm_net, num_components=num_components, u_min=-10., u_max=10.)

        self.encoder = Encoder()
        self.decoder = Decoder(num_vals=num_vals, distribution=distribution)
        self.num_vals = num_vals
        self.latent_dim = latent
    def loss_function(self,batch_data,encoder_model,decode_model):
        """
        .. math:: L = {q(z|x)}[\ln p(x|z)] + KL(q(z|x)||p(z))
        >>> loss = -log_p + KL
        >>> KL = 0.5 * torch.sum(mu_e ** 2 + torch.exp(log_var_e) - log_var_e - 1)
        :param parm_b: int
        :param parm_c: int
        :return:
        """
        mu_e, log_var_e = encoder_model(batch_data)
        z = self.encoder.sample(mu_e, log_var_e)
        #ELBO
        outs = decode_model(z)
        RE = self.decoder.log_prob(batch_data, outs)
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)
        return -(RE + KL).mean()

    def sample(self, decode_model,batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(decode_model,z)

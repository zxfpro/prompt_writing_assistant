from layertools import log_normal_diag,log_categorical,log_bernoulli
import torch
import torch.nn as nn
import torch.nn.functional as F


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

# 假设隐空间分布
class VampPrior():
    def __init__(self, L, D, num_vals, num_components, data=None):
        self.L = L
        self.D = D
        self.num_vals = num_vals

        # pseudoinputs
        u = torch.rand(num_components, D) * self.num_vals
        self.u = nn.Parameter(u)

        # mixing weights
        self.w = nn.Parameter(torch.zeros(self.u.shape[0], 1, 1)) # K x 1 x 1

    def get_params(self,encoder_model):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = encoder_model(self.u) #(K x L), (K x L)
        return mean_vampprior, logvar_vampprior

    def sample(self, batch_size, encoder_model):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params(encoder_model)

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

    def log_prob(self, z, encoder_model):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params(encoder_model) # (K x L) & (K x L)

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
    def __init__(self, L, D, gtm_net, num_points, u_min=-10., u_max=10., num_vals=255):
        super(GTMVampPrior, self).__init__()

        self.L = L
        self.D = D
        self.num_vals = num_vals


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

    def get_params(self,encoder_model):
        # u->gtm_net->u_x
        h_gtm = self.gtm_net(self.u) #K x D
        h_gtm = h_gtm * self.num_vals
        # u_x->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = encoder_model(h_gtm) #(K x L), (K x L)
        return mean_vampprior, logvar_vampprior

    def sample(self, batch_size,encoder_model):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params(encoder_model)

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

    def log_prob(self, z,encoder_model):
        # u->encoder->mu, lof_var
        mean_vampprior, logvar_vampprior = self.get_params(encoder_model)

        # mixing probabilities
        w = F.softmax(self.w, dim=0)

        # log-mixture-of-Gaussians
        z = z.unsqueeze(0) # 1 x B x L
        mean_vampprior = mean_vampprior.unsqueeze(1) # K x 1 x L
        logvar_vampprior = logvar_vampprior.unsqueeze(1) # K x 1 x L

        log_p = log_normal_diag(z, mean_vampprior, logvar_vampprior) + torch.log(w) # K x B x L
        log_prob = torch.logsumexp(log_p, dim=0, keepdim=False) # B x L

        return log_prob


class Process():
    """
    ().
    """
    def __init__(self,latent,num_vals= None,D=None,distribution = 'categorical',type = 'gtm_vamp'):
        if type =='vamp':
            num_components = 4 ** 2
            self.prior = VampPrior(L=latent, D=D, num_vals=num_vals,
                                   num_components=num_components)
        elif type == 'gtm_vamp':
            num_components = 4
            gtm_net_vamp = nn.Sequential(nn.Linear(2, 256), nn.Tanh(),
                                         nn.Linear(256, 256), nn.Tanh(),
                                         nn.Linear(256, D), nn.Sigmoid())

            self.prior = GTMVampPrior(L=latent, D=D, gtm_net=gtm_net_vamp, num_points=num_components,
                                      u_min=-10., u_max=10., num_vals=num_vals)

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
        KL = (self.prior.log_prob(z,encoder_model) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)
        return -(RE + KL).mean()

    def sample(self, decode_model,encode_model,batch_size=64):
        z = self.prior.sample(batch_size=batch_size,encoder_model=encode_model)
        return self.decoder.sample(decode_model,z)


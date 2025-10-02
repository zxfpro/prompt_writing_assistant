from layertools import log_normal_diag,log_categorical,log_bernoulli
import torch

# import logging
# # 设置logging level
# logging.basicConfig(level=logging.DEBUG)
import logging
logger = logging.getLogger('process')
logger.setLevel(logging.WARNING)

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

class StandardPrior():
    # 有一个大大的疑问 ,那么这个东东和上面的Prior有什么区别呢
    def __init__(self, L=2):
        self.L = L
        # params weights
        self.means = torch.zeros(1, L)
        self.logvars = torch.zeros(1, L)

    def get_params(self):
        return self.means, self.logvars

    def sample(self, batch_size):
        return torch.randn(batch_size, self.L)

    def log_prob(self, z):
        """
        表示Ln(p(x))
        """
        zero = torch.Tensor([0])
        return log_normal_diag(z, zero, zero)

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
            logger.debug('categorical')
            mu_d = outs[0]
            logger.debug(mu_d.shape)
            logger.debug(x.shape)

            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.distribution == 'bernoulli':
            logger.debug('bernoulli')
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')
        return log_p

class Process():
    """
    ().
    """
    def __init__(self,latent,num_vals= None,distribution = 'categorical'):
        self.prior = Prior(latent)
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
        logger.info('1')
        logger.debug(batch_data.shape)
        mu_e, log_var_e = encoder_model(batch_data)

        z = self.encoder.sample(mu_e, log_var_e)
        #ELBO
        outs = decode_model(z)
        logger.info('2')
        logger.debug('outs:{}'.format(outs))
        RE = self.decoder.log_prob(batch_data, outs)
        logger.info('3')
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)
        logger.info('4')
        return -(RE + KL).mean()

    def sample(self, decode_model,batch_size=64):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(decode_model,z)


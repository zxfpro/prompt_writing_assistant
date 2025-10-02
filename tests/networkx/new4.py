```



import torch
import torch.nn as nn

class ScoreNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden=64, output_dim=1, c_dim=64, nb_timesteps=200):
        super().__init__()

        dims = torch.arange(c_dim // 2).unsqueeze(0)  # (1, c_dim  // 2)
        steps = torch.arange(nb_timesteps).unsqueeze(1)  # (nb_timesteps, 1)
        first_half = torch.sin(steps * 10. ** (dims * 4. / (c_dim // 2 - 1)))
        second_half = torch.cos(steps * 10. ** (dims * 4. / (c_dim // 2 - 1)))
        diff_embedding = torch.cat((first_half, second_half), dim=1)  # (nb_timesteps, c_dim)
        self.register_buffer('diff_embedding', diff_embedding)

        self.init = nn.Conv1d(input_dim, hidden, 1)
        self.init_skip = nn.Conv1d(input_dim, hidden, 1)
        # elif type == 2:
        #     self.init = nn.Conv2d(input_dim, hidden, 1)
        #     self.init_skip = nn.Conv2d(input_dim, hidden, 1)
        # 28x28
        self.l1 = ConvGLU(hidden, c_dim)
        self.down1 = DownsampleBlock(hidden)

        # 14x14
        self.l2 = ConvGLU(hidden * 2, c_dim)
        self.down2 = DownsampleBlock(hidden * 2)

        # 7x7
        self.l3 = ConvGLU(hidden * 4, c_dim, dilation=1)
        self.l4 = ConvGLU(hidden * 4, c_dim, dilation=2)

        # 14x14
        self.up1 = UpsampleBlock(hidden * 4)
        self.l5 = ConvGLU(hidden * 2, c_dim)

        # 28x28
        self.up2 = UpsampleBlock(hidden * 2)
        self.l6 = ConvGLU(hidden, c_dim)

        self.rescale = 0.5 ** 0.5

        self.out = nn.Sequential(
            nn.Conv1d(hidden, hidden * 2, 1),
            nn.LeakyReLU(),
            nn.Conv1d(hidden * 2, 1, 1)
        )
        # elif type == 2:
        #     self.out = nn.Sequential(
        #         nn.Conv2d(hidden, hidden * 2, 1),
        #         nn.LeakyReLU(),
        #         nn.Conv2d(hidden * 2, 1, 1)
        #     )

    def forward(self, x, t):
        """
        x为sample_q出来的 x图  t为步长
        Produces an estimate for the noise term `epsilon`.
        Can equivalently be thought of as a learned gradient or "score function" estimator,
        as part of a Langevin sampling procedure.
        :param x: (batch, 1, H, W) torch.float
        :param t: (batch) torch.int
        """

        batch = x.shape[0]
        c = self.diff_embedding[t]  # (batch, c_dim)
        # initial channel upsampling
        out = self.init(x)
        skip = self.init_skip(x)

        # 28x28
        out, skip = self.l1(out, skip, c)
        out_half, skip_half = self.down1(out, skip)

        # 14x14
        out_half, skip_half = self.l2(out_half, skip_half, c)
        out_quart, skip_quart = self.down2(out_half, skip_half)

        # 7x7
        out_quart, skip_quart = self.l3(out_quart, skip_quart, c)
        out_quart, skip_quart = self.l4(out_quart, skip_quart, c)

        # back to 14x14
        out_half_, skip_half_ = self.up1(out_quart, skip_quart)

        out_half = self.rescale * (out_half + out_half_)
        skip_half = self.rescale * (skip_half + skip_half_)
        # 14x14
        out_half, skip_half = self.l5(out_half, skip_half)

        # back to 28x28
        out_, skip_ = self.up2(out_half, skip_half)
        out = self.rescale * (out + out_)
        skip = self.rescale * (skip + skip_)

        out, skip = self.l6(out, skip, c)
        out = self.out(skip)
        return out

class DownsampleBlock(nn.Module):
    """
    下采样 尺寸减半 通道翻倍
    """
    def __init__(self, hidden ):
        super(DownsampleBlock, self).__init__()

        self.main = nn.Conv1d(hidden, hidden * 2, kernel_size=2, stride=2)
        self.skip = nn.Conv1d(hidden, hidden * 2, kernel_size=2, stride=2)
        # elif type = =2:
        #     self.main = nn.Conv2d(hidden, hidden * 2, kernel_size=2, stride=2)
        #     self.skip = nn.Conv2d(hidden, hidden * 2, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.main(x)
        skip = self.skip(skip)
        return x, skip

class UpsampleBlock(nn.Module):
    """
    上采样 尺寸翻倍 通道减半
    """
    def __init__(self, hidden ):
        super(UpsampleBlock, self).__init__()

        self.main = nn.ConvTranspose1d(hidden, hidden // 2, kernel_size=2, stride=2)
        self.skip = nn.ConvTranspose1d(hidden, hidden // 2, kernel_size=2, stride=2)
        # elif type ==2:
        #     self.main = nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=2, stride=2)
        #     self.skip = nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.main(x)
        skip = self.skip(skip)
        return x, skip

class ConvGLU(nn.Module):
    """
    结合了Res残差块
    padding只有1,2 根据dilation来的
    在没有c的情况下,x和skip 分别进行了res块的构建
    """

    def __init__(self, channels, c_dim=None, dilation=1, kw=3,):
        """
        up1x1 通道数翻倍
        skip1x1 通道数翻倍
        conv 通道不变 卷积核为3 dilation为2时是扩散卷积
        c_proj 为一个c_dim 到 channel 的全连接层
        """
        super().__init__()

        self.bn = nn.BatchNorm1d(channels)
        # elif type == 2:
        #     self.bn = nn.BatchNorm2d(channels)

        # main op. + parameterised residual connection
        padding = (kw - 1) * dilation // 2

        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=padding)
        self.up1x1 = nn.Conv1d(channels // 2, channels, 1)
        # parameterised skip connection
        self.skip1x1 = nn.Conv1d(channels // 2, channels, 1)

        # elif type == 2:
        #     self.conv = nn.Conv2d(channels, channels, kernel_size=3, dilation=dilation, padding=padding)
        #     self.up1x1 = nn.Conv2d(channels // 2, channels, 1)
        #     # parameterised skip connection
        #     self.skip1x1 = nn.Conv2d(channels // 2, channels, 1)

        self.rescale = 0.5 ** 0.5

        if c_dim is not None:
            self.c_proj = nn.Linear(c_dim, channels)

    def forward(self, x, skip, c=None ):
        """
        :param x: (B, C H, W)
        :param skip: (B, C, H, W)
        """
        h = self.bn(x)
        h = self.conv(h)
        a, b = torch.chunk(h, 2, 1)  # (B, C // 2, H, W)

        if c is not None:
            assert hasattr(self, 'c_proj'), "Oops, conditioning dim not specified!"
            batch = x.shape[0]
            c_proj = self.c_proj(c)
            c_a, c_b = torch.chunk(c_proj, 2, -1)  # (B, C // 2)
            c_a = c_a.reshape(batch, -1, 1)  # (B, C // 2, H=1, W=1)
            c_b = c_b.reshape(batch, -1, 1)
            # elif type == 2:
            #     c_a = c_a.reshape(batch, -1, 1, 1)  # (B, C // 2, H=1, W=1)
            #     c_b = c_b.reshape(batch, -1, 1, 1)
            a = (a + c_a) * self.rescale
            b = (b + c_b) * self.rescale

        out = torch.sigmoid(a) * b

        # accumulate skip values
        skip = self.rescale * (self.skip1x1(out) + skip)  # (B, C, H, W)

        # residual connection
        out = self.up1x1(out)
        out = self.rescale * (out + x)  # (B, C, H, W)

        return out, skip


import torch
from gmodel.base import NetworkBase
from typing import *
from zxfMLtools.models.diffusion_DDPM.nn_ import ScoreNetwork
from ..utils import check_nnseq

class Network(NetworkBase):
    data_demand = None
    def __init__(self, input_size:Tuple, nnseq:Callable[[torch.Tensor],object] or None=None ,
                 dim_channel_list=[20, 40, 60],
                 nb_timesteps=200,hidden=64, c_dim=64):

        if nnseq is None:
            network_obj = ScoreNetwork(input_dim=1,
                         hidden=hidden, output_dim=1,
                         c_dim=c_dim, nb_timesteps=nb_timesteps)
        else:
            network_obj = nnseq
            check_nnseq(input_size=input_size, output_size=(input_size), nn_obj=network_obj)

        super().__init__(input_size=input_size,dim_channel_list=dim_channel_list,
                                           network=network_obj
                                           )
        self.nb_timesteps = nb_timesteps
        self.hidden = hidden
        self.c_dim = c_dim

    def forward(self, batch):
        return self.network(batch)

    def get_information(self):
        return super().get_information() + f'''
nb_timesteps    ：{self.nb_timesteps}
hidden          : {self.hidden }
c_dim           : {self.c_dim}
'''










import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from zxfMLtools.models.base import ModelBase, NetworkBase


class DiffusionProcess(nn.Module):
    def __init__(self, nb_timesteps=50, start=1e-4, end=0.05):
        super().__init__()

        self.nb_timesteps = nb_timesteps

        # beta = likelihood variance q(x_t | x_t-1)
        beta = torch.linspace(start, end, nb_timesteps)
        alpha = 1. - beta
        alpha_hat = alpha.cumprod(dim=0)

        # q(x_t|x_0) = N(x_t ; sqrt(alpha_hat) * x_0, forward_variance)
        prior_variance = (1. - alpha_hat)

        # forward process posterior variance (beta_hat) corresponding to (q(x_t-1 | x_t, x_0)
        alpha_hat_t_1 = F.pad(alpha_hat, (1, 0))[:-1]
        posterior_variance = (1 - alpha_hat_t_1) * beta / (1 - alpha_hat)
        posterior_variance[0] = beta[0]

        for (name, tensor) in [
            ("beta", beta),
            ("alpha", alpha),
            ("alpha_hat", alpha_hat),
            ("prior_variance", prior_variance),
            ("posterior_variance", posterior_variance)
        ]:
            self.register_buffer(name, tensor)

    def sample_t(self, batch_size=1, device=None):
        """
        随机生成一个常数(每个batch一个)范围就是0-nb_time
        为每一个批次随机取样一个时间步长
        Sample a random timestep for each batch item.
        """
        return torch.randint(0, self.nb_timesteps, size=(batch_size,), device=device)

    def sample_q(self, x0, eps, t, device='cpu'):
        """
        x0就是原始图片
        eps就是高斯噪声
        t就是当前时刻
        生成 就是q(xt|x0) 来的 xt的分布
        The "forward process". Given the data point x_0, we can sample
        any latent x_t from q(x_t|x_0)
        :param x0: the initial data point (batch, *)
        :param eps: noise samples from N(0, I) (batch, *)
        :param t: the timesteps in [0, nb_timesteps] (batch)
        """
        assert (t >= 0).all() and (t < self.nb_timesteps).all(), "Invalid timestep"

        alpha_hat_t = self.alpha_hat[t, None, None]  # (batch, 1, 1)
        # elif self.type == 2:
        #     alpha_hat_t = self.alpha_hat[t, None, None, None]  # (batch, 1, 1, 1)
        alpha_hat_t = alpha_hat_t.to(device)
        return alpha_hat_t.sqrt() * x0 + (1. - alpha_hat_t).sqrt() * eps

    def sample_p(self, x_t, eps_hat, t, greedy=False, device='cpu'):
        """
        输入的x是高斯噪声,esp是生成的分布 t_ 是时间步(经过维度调整)
        #生成图片

        The "reverse process". Given a latent `x_t`, draw a sample from p(x_t-1|x_t)
        using the noise prediction.
        :param x_t: the previous sample (batch, *)
        :param eps_hat: the noise, predicted by neural net (batch, *)
        :param t: the timestep (batch)
        """
        alpha_t = self.alpha[t, None, None]  # (batch, 1, 1, 1)
        beta_t = self.beta[t, None, None]
        alpha_hat_t = self.alpha_hat[t, None, None]
        # elif self.type == 2:
        #     alpha_t = self.alpha[t, None, None, None]  # (batch, 1, 1, 1)
        #     beta_t = self.beta[t, None, None, None]
        #     alpha_hat_t = self.alpha_hat[t, None, None, None]

        alpha_hat_t = alpha_hat_t.to(device)
        alpha_t = alpha_t.to(device)
        beta_t = beta_t.to(device)
        # calculate the mean
        mu = x_t - ((beta_t * eps_hat) / (1. - alpha_hat_t).sqrt())
        mu = (1. / alpha_t.sqrt()) * mu

        if greedy:
            return mu

        # sample
        std = self.posterior_variance[t, None, None].sqrt()
        # elif self.type == 2:
        #     std = self.posterior_variance[t, None, None, None].sqrt()
        std = std.to(device)
        x_next = mu + std * torch.randn_like(mu).to(device)

        return x_next


class Model(ModelBase):
    name = 'diffusion'
    def __init__(self, network: NetworkBase, lr, once_num=10, start=1e-4, end=0.05, device='cpu'):
        optimizer = optim.Adam(network.network.parameters(),lr=lr)
        super().__init__(optimizer=optimizer, lr=lr,device=device)
        self.network = network.to(device)
        self.diffusion = DiffusionProcess(nb_timesteps=self.network.nb_timesteps, start=start, end=end)
        self.once_num = once_num  # 一次生成的数量，防止爆内存
        self.start = start
        self.end = end

    def forward(self, x, cond=None):
        x = x.reshape(x.shape[0], 1, -1).float()
        x = x.to(self.device)
        eps = torch.randn_like(x, device=self.device)
        t = self.diffusion.sample_t(eps.shape[0], device=self.device)
        x_t = self.diffusion.sample_q(x, eps, t, device=self.device)
        eps_hat = self.network.network(x_t, t)
        return eps_hat, eps

    def loss_function(self, eps_hat, eps):
        return F.mse_loss(eps_hat, eps, reduction='mean')

    def train(self, batch_data, cond=None):
        self.optimizer.zero_grad()
        eps_hat, eps = self.forward(batch_data, cond)
        loss = self.loss_function(eps_hat, eps)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _generate(self, shape):
        batch, *_ = shape
        x_T = torch.randn(shape, device=self.device)
        x = x_T
        out = [x_T]

        for t in range(self.diffusion.nb_timesteps - 1, -1, -1):
            t_ = torch.full((batch,), t, device=self.device)

            eps_hat = self.network.network(x, t_)
            x = self.diffusion.sample_p(x, eps_hat, t_, greedy=t == 0, device=self.device)
            if (t + 1) % 20 == 0:
                out.append(x)
        return x, out

    def generate(self, batch_size=100, cond=None):
        generated_data = torch.Tensor().to(self.device)
        for i in range(int(batch_size / self.once_num)):
            x_gen, _ = self._generate((self.once_num, 1, *self.network.input_size))
            generated_data_cell = x_gen.reshape(self.once_num, -1).detach()
            generated_data = torch.cat([generated_data, generated_data_cell], dim=0)

            del x_gen, _
            torch.cuda.empty_cache()
        return generated_data

    def get_information(self):
        return super().get_information() + f"""one_num : {self.once_num}
start : {self.start}  
end : {self.end}  
"""






```


```


```
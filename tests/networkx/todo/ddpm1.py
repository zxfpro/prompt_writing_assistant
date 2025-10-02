
from zxfMLtools.models.ddpm.network.network import CNetwork
from .ddpm1 import ScoreNetwork


def Network(name='c',**kwargs):
    return CNetwork(**kwargs) if name == 'c' else ScoreNetwork(**kwargs)


import torch.nn as nn
import torch.nn.functional as F
from diffusion.base import NetworkBase


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, y):
        out = self.lin(x)
        gamma = self.embed(y)
        out = gamma.view(-1, self.num_out) * out
        return out

class CNetwork(NetworkBase):
    """
    n_steps

    """
    def __init__(self, n_steps,input_size):
        super(CNetwork, self).__init__()
        self.n_timesteps = n_steps
        self.lin1 = ConditionalLinear(input_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, input_size)

    def forward(self, x, y):
        x = F.softplus(self.lin1(x, y))
        x = F.softplus(self.lin2(x, y))
        x = F.softplus(self.lin3(x, y))
        return self.lin4(x)

import torch
import torch.nn as nn

class DownsampleBlock(nn.Module):
    """
    下采样 尺寸减半 通道翻倍
    """
    def __init__(self, hidden):
        super(DownsampleBlock, self).__init__()

        self.main = nn.Conv2d(hidden, hidden * 2, kernel_size=2, stride=2)
        self.skip = nn.Conv2d(hidden, hidden * 2, kernel_size=2, stride=2)

    def forward(self, x, skip):
        x = self.main(x)
        skip = self.skip(skip)
        return x, skip

class UpsampleBlock(nn.Module):
    """
    上采样 尺寸翻倍 通道减半
    """
    def __init__(self, hidden):
        super(UpsampleBlock, self).__init__()

        self.main = nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=2, stride=2)
        self.skip = nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=2, stride=2)

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
    def __init__(self, channels, c_dim=None, dilation=1, kw=3):
        """
        up1x1 通道数翻倍
        skip1x1 通道数翻倍
        conv 通道不变 卷积核为3 dilation为2时是扩散卷积
        c_proj 为一个c_dim 到 channel 的全连接层
        """
        super().__init__()

        self.bn = nn.BatchNorm2d(channels)

        # main op. + parameterised residual connection
        padding = (kw - 1) * dilation // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, dilation=dilation, padding=padding)
        self.up1x1 = nn.Conv2d(channels // 2, channels, 1)

        # parameterised skip connection
        self.skip1x1 = nn.Conv2d(channels // 2, channels, 1)

        self.rescale = 0.5 ** 0.5

        if c_dim is not None:
            self.c_proj = nn.Linear(c_dim, channels)

    def forward(self, x, skip, c=None):
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
            c_a = c_a.reshape(batch, -1, 1, 1)  # (B, C // 2, H=1, W=1)
            c_b = c_b.reshape(batch, -1, 1, 1)
            a = (a + c_a) * self.rescale
            b = (b + c_b) * self.rescale

        out = torch.sigmoid(a) * b

        # accumulate skip values
        skip = self.rescale * (self.skip1x1(out) + skip)  # (B, C, H, W)

        # residual connection
        out = self.up1x1(out)
        out = self.rescale * (out + x)  # (B, C, H, W)

        return out, skip

class ScoreNetwork(nn.Module):
    """
    就是一个先压缩后扩散的一个网络,
    那这个网络输入输出的不同在于  宽度,高度都是相同的,维度压缩到了一维 这么看 维度也是相同的
    也就是说,输入输出前后的图片尺寸没有变化
    """
    def __init__(self, input_dim=3, hidden=64, output_dim=3, c_dim=64, nb_timesteps=50):
        super().__init__()

        # create positional embeddings (Vaswani et al, 2018)
        dims = torch.arange(c_dim // 2).unsqueeze(0)  # (1, c_dim  // 2)
        steps = torch.arange(nb_timesteps).unsqueeze(1)  # (nb_timesteps, 1)
        first_half = torch.sin(steps * 10. ** (dims * 4. / (c_dim // 2 - 1)))
        second_half = torch.cos(steps * 10. ** (dims * 4. / (c_dim // 2 - 1)))
        diff_embedding = torch.cat((first_half, second_half), dim=1)  # (nb_timesteps, c_dim)
        self.register_buffer('diff_embedding', diff_embedding)

        self.init = nn.Conv2d(input_dim, hidden, 1)
        self.init_skip = nn.Conv2d(input_dim, hidden, 1)

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
            nn.Conv2d(hidden, hidden * 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden * 2, 1, 1)
        )

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


import torch
import torch.nn.functional as F
class Process:
    """
     n_timesteps,
     beta_type='linear',"quad","sigmoid"
     start=1e-5,
     end=1e-2
    """

    def __init__(self,n_timesteps,beta_type='linear',start=1e-5,end=1e-2):
        self.n_timesteps = n_timesteps
        betas = self._make_beta_schedule(schedule=beta_type,n_timesteps=n_timesteps,start=start,end=end)
        self.alphas = 1. - betas
        self.alphas_hat = self.alphas.cumprod(dim=0)
        self.alphas_hat_sqrt = self.alphas_hat.sqrt_()
        self.one_minus_alphas_hat_sqrt = (1. - self.alphas_hat).sqrt_()

    def sample_q(self,x_0, t, noise=None):
        """
        前向过程 数字化
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        alphas_t = self.extract(self.alphas_hat_sqrt, t, x_0)
        alphas_1_m_t = self.extract(self.one_minus_alphas_hat_sqrt, t, x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)


    def sample_p(self,model,x_t, t,):
        eps_factor = (1 - self.extract(self.alphas, t, x_t)) / self.extract(self.one_minus_alphas_hat_sqrt, t, x_t)
        eps_theta = model(x_t, t)
        mean = (1/self.extract(self.alphas, t, x_t).sqrt()) * (x_t - (eps_factor * eps_theta))
        z = torch.randn_like(x_t)
        sigma_t = self.extract((1. - self.alphas), t, x_t).sqrt()
        sample = mean + sigma_t * z
        return sample

    def sample_t(self,batch_size=1,device='cpu'):
        t = torch.randint(0, self.n_timesteps, size=(batch_size,), device=device)
        return t

    def loss_function(self, eps_hat, eps):
        return F.mse_loss(eps_hat, eps, reduction='mean')


    def extract(self,input, t, x):
        """
        一个未知的结构转换方法
        """
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def _make_beta_schedule(self,schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        """
        获取一个β列表 shape=[n_timesteps] 起始点为start 终点为end
        提供三种模式,线性 linear 指数 quad 激活 sigmoid
        """
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas




import torch
import torch.optim as optim
from zxfMLtools.models.base import ModelBase,NetworkBase
from zxfMLtools.models.ddpm.process import Process


class Model(ModelBase):
    """
    Model(network:NetworkBase,
                 lr:float,
                 process:DiffusionProcess,
                 device='cpu')

    Modle.train
    Model.generate
    """
    def __init__(self, network:NetworkBase,
                 lr:float,
                 process:Process, device='cpu'):

        super().__init__()
        try:
            assert network.n_timesteps == process.n_timesteps
        except AssertionError:
            raise AssertionError('network and process must have the same number of timesteps')

        self.network = network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.process = process
        self.device = device

    def forward(self, batch_data, cond=None) -> (list, dict):
        eps = torch.randn_like(batch_data, device=self.device)
        t = self.process.sample_t(eps.shape[0], self.device)
        x_0 = batch_data.to(self.device)
        x_t = self.process.sample_q(x_0,t,eps)
        eps_hat = self.network(x_t, t)
        return eps_hat, eps

    def train_epoch(self, batch_data, cond=False):
        """
        batch_data:torch.Tensor
        cond:torch.Tensor

        return loss
        """
        self.optimizer.zero_grad()
        forward_param = self.forward(batch_data, cond)
        loss = self.process.loss_function(*forward_param)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def lightning_train(self, batch_data, cond=False):
        forward_param = self.forward(batch_data, cond)
        loss = self.process.loss_function(*forward_param)
        return loss

    def generate(self, batch_size=1, cond=None):
        #TODO
        shape = self.network.in_shape
        x_t = torch.randn(shape, device=self.device)
        x_list = [x_t]
        for t in reversed(range(self.network.n_timesteps)):
            x_t = self.process.sample_p(self.network, x_t, t)
            x_list.append(x_t)
        return x_list

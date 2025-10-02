import torch
import torch.nn as nn
import torch.optim as optim
from typing import *
import torch
import torch.nn as nn
import torch
class ModelBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_data, cond=None) -> (list, dict):
        # 用于前向传播
        raise NotImplemented

    def loss_function(self, *args, **kwargs) -> torch.Tensor:
        # 用于损失函数
        raise NotImplemented

    def train(self, batch_data, cond=False) -> int:
        # 用于训练过程
        raise NotImplemented

    def generate(self, batch_size: int, cond=None):
        raise NotImplemented


class NetworkBase(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplemented


class Network(NetworkBase):
    data_demand = ['column', 'cond', 'scatter']

    def __init__(self, input_size:Tuple, nnseq:Callable[[torch.Tensor],object] or None=None ,
                 dim_channel_list:List=[30, 40, 60],
                 scatter_dim=32):

        if nnseq is None:
            network_obj = set_linear(input_size = input_size,output_size = scatter_dim,
                                     dim_channel_list=dim_channel_list)
            # def set_linear(input_size, dim_channel_list, scatter_dim):
            #     in_channel = np.prod(input_size)
            #     network = nn.Sequential(View(),
            #                             MLP([in_channel, *dim_channel_list, scatter_dim], shape_len=len(input_size),
            #                                 active='elu'))
            #     return network
            # network = set_linear(input_size, dim_channel_list)
        else:
            network_obj = nnseq
            check_nnseq(input_size=input_size,output_size=(*input_size, scatter_dim),nn_obj=network_obj)

        super().__init__(input_size=input_size,
                                     dim_channel_list=dim_channel_list,
                                     network=network_obj)
        self.scatter_dim = scatter_dim

    def forward(self, batch):
        return self.network(batch)

    def get_information(self):
        return super().get_information() + f"""scatter_dim :     {self.scatter_dim}
"""



class Model(ModelBase):
    name = 'EBM'
    def __init__(self, network: NetworkBase, lr, ld_steps=20, alpha=1.0, sigma=0.01, device='cpu'):
        optimizer = optim.Adam(network.parameters(), lr=lr),
        super().__init__(optimizer = optimizer,lr=lr,device=device)
        self.network = network.to(device)
        self.nll = nn.NLLLoss(reduction='none')
        self.ld_steps = ld_steps
        self.alpha = torch.FloatTensor([alpha])
        self.sigma = sigma

    def forward(self, batch_data, cond=None):
        y = self.network.network(batch_data)
        return y

    def loss_function(self, x, cond, y):
        loss_clf = self.class_loss(y, cond)
        loss_gen = self.gen_loss(x, y)
        loss = (loss_clf + loss_gen).mean()
        self.loss_clf = loss_clf.mean()
        self.loss_gen = loss_gen.mean()
        return loss

    def class_loss(self, y, cond):
        y_pred = torch.softmax(y, 1)
        # cond = cond.reshape(-1)#TODO
        # cond = cond.long()

        return self.nll(torch.log(y_pred), cond)

    def gen_loss(self, x, y):
        x_sample = self.generate(batch_size=x.shape[0])
        # - calculate f(x_sample)[cond]
        y_sample = self.network.network(x_sample)
        return -(torch.logsumexp(y, 1) - torch.logsumexp(y_sample, 1))

    def train(self, batch_data, cond=None):
        self.optimizer.zero_grad()
        y = self.forward(batch_data, cond)
        loss = self.loss_function(batch_data,cond,y)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item(), self.loss_clf.item(), self.loss_gen.item()

    def energy_gradient(self, x):
        # dataset original data that doesn't require grads!

        # x_i = torch.FloatTensor(x.data)
        x_i = x.data
        x_i.requires_grad = True  # WE MUST ADD IT, otherwise autograd won't syn_2
        # calculate the gradient
        x_i.type_as(x)
        x_i_grad = torch.autograd.grad(torch.logsumexp(self.network.network(x_i), 1).sum(), [x_i], retain_graph=True)[0]
        # self.train()
        return x_i_grad

    def langevin_dynamics_step(self, x_old, alpha):
        # Calculate gradient wrt x_old
        grad_energy = self.energy_gradient(x_old)
        # Sample eta ~ Normal(0, alpha)
        epsilon = torch.randn_like(grad_energy) * self.sigma
        # New sample
        alpha = alpha.type_as(x_old)
        x_new = x_old + alpha * grad_energy + epsilon
        return x_new

    def generate(self, batch_size, cond=None):
        # - 1) Sample from uniform
        x_sample = 2. * torch.rand([batch_size, self.network.in_channel], device=self.device) - 1.
        # - 2) run Langevine Dynamics
        for i in range(self.ld_steps):
            x_sample = self.langevin_dynamics_step(x_sample, alpha=self.alpha)
        return x_sample

    def get_information(self):
        return super().get_information() + \
f"""
ld_steps:     {self.ld_steps}
alpha:        {self.alpha}
sigma:        {self.sigma}
"""
from zxfMLtools.models.ebms.ebmodels.base import NetworkBase
from layertools import set_linear
"""
origin.Model
origin.Network
origin.EBMProcess
"""

from .ebm_model2 import Model
from zxfMLtools.models.ebms.ebmodels.origin.network import Network
from zxfMLtools.models.ebms.ebmodels.origin.process import Process

class Network(NetworkBase):
    def __init__(self,input_size,num_vals):
        super(Network, self).__init__()
        self.num_vals = num_vals
        self.net1 = set_linear(input_size = input_size,output_size=(*input_size, num_vals),
                                     dim_channel_list=[30, 40, 60])

    def forward(self, batch_data):
        return self.net1(batch_data)

import torch
import torch.nn as nn

class Process:
    """
    EBMProcess(self, sigma=0.01)

    """
    def __init__(self, sigma=0.01):
        self.nll = nn.NLLLoss(reduction='none')
        self.sigma = sigma

    def loss_function(self, x, cond, y):
        print(1)
        loss_clf = self.class_loss(y, cond)
        print(2)
        loss_gen = self.gen_loss(x, y)
        print(3)
        return (loss_clf + loss_gen).mean()

    def class_loss(self, y, cond):
        print(y.shape,'y.shape')
        y_pred = torch.softmax(y, 1)
        # cond = cond.reshape(-1)#TODO
        # cond = cond.long()
        print(y_pred.shape)
        print(cond.shape,'cond.shape')
        return self.nll(torch.log(y_pred), cond)

    def gen_loss(self, x, y):
        x_sample = self.generate(batch_size=x.shape[0])
        # - calculate f(x_sample)[cond]
        y_sample = self.network.network(x_sample)
        return -(torch.logsumexp(y, 1) - torch.logsumexp(y_sample, 1))

    def langevin_dynamics_step(self, x_old, network, alpha):
        # Calculate gradient wrt x_old
        grad_energy = self.energy_gradient(network, x_old)
        # Sample eta ~ Normal(0, alpha)
        epsilon = torch.randn_like(grad_energy) * self.sigma
        # New sample
        alpha = alpha.type_as(x_old)
        x_new = x_old + alpha * grad_energy + epsilon
        return x_new

    def energy_gradient(self, network, x):
        # dataset original data that doesn't require grads!
        # x_i = torch.FloatTensor(x.data)
        x_i = x.data
        x_i.requires_grad = True  # WE MUST ADD IT, otherwise autograd won't syn_2
        # calculate the gradient
        x_i.type_as(x)
        x_i_grad = torch.autograd.grad(torch.logsumexp(network(x_i), 1).sum(), [x_i], retain_graph=True)[0]
        # self.train()
        return x_i_grad

import torch
import torch.optim as optim
from zxfMLtools.models.ebms.ebmodels.origin.process import Process
from zxfMLtools.models.ebms.ebmodels.base import ModelBase,NetworkBase
class Model(ModelBase):
    """
    models.train
    models.generate

    """
    def __init__(self, network:NetworkBase,
                 lr:float,
                 sigma:float=0.01,
                 device='cpu'):
        super().__init__()
        self.network = network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.process = Process(sigma=sigma)
        self.device = device

    def forward(self, batch_data, cond=None):
        y = self.network(batch_data)
        return y

    def train_epoch(self, batch_data, cond=None):
        self.optimizer.zero_grad()
        y = self.forward(batch_data, cond)
        loss = self.process.loss_function(batch_data, cond, y)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()

    def lightning_train(self, batch_data, cond=False):
        # 用于训练过程
        y = self.forward(batch_data, cond)
        loss = self.process.loss_function(batch_data, cond, y)
        return loss

    def generate(self, batch_size, cond=None,alpha = 1.0,ld_steps = 20,):
        # - 1) Sample from uniform
        x_sample = 2. * torch.rand([batch_size, self.network.in_channel], device=self.device) - 1.
        # - 2) run Langevine Dynamics
        for i in range(ld_steps):
            x_sample = self.process.langevin_dynamics_step(x_sample, alpha=torch.FloatTensor([alpha]))
        return x_sample



import torch
import torch.nn.functional as F
import torch.optim as optim
from ..base import NetworkBase, ModelBase

EPS = 1.e-5


class Model(ModelBase):
    """
    network: Callable 网络结构
    lr : float 学习率
    device : str 设备

    """
    name = 'ARM'
    def __init__(self, network: NetworkBase, lr, device='cpu'):
        optimizer = optim.Adam(network.parameters(), lr=lr)
        super().__init__(optimizer=optimizer, lr=lr, device=device)
        self.network = network.to(device)

    def forward(self, batch_data, cond=None):
        mu_d = self.network(batch_data)
        return mu_d

    def loss_function(self, x, mu_d, dim=-1):
        x_one_hot = F.one_hot(x.long(), num_classes=self.network.scatter_dim)
        log_p = x_one_hot * torch.log(torch.clamp(mu_d, EPS, 1. - EPS))
        return -torch.mean(log_p, dim).sum(-1).mean()

    def train(self, batch_data, cond=None):
        self.optimizer.zero_grad()
        x = batch_data
        mu_d = self.forward(batch_data, cond)
        loss = self.loss_function(x, mu_d)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def generate(self, batch_size, cond=None):
        window_size = self.network.input_size[-1]
        x_new = torch.zeros((batch_size, window_size), device=self.device)
        for d in range(window_size):
            _, p = self.forward(x_new.unsqueeze(1), cond=None)
            x_new_d = torch.multinomial(p[:, d, :], num_samples=1)
            x_new[:, d] = x_new_d[:, 0]
        return x_new

    def get_information(self):
        return super().get_information() + """"""


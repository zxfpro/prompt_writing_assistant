import torch
import torch.nn.functional as F
from vae.dele.vaebase import VAEModelBase,VAENetworkBase

EPS = 1.e-5

class Model(VAEModelBase):
    name = 'vae_priors'

    def __init__(self, network: VAENetworkBase, lr, device='cpu'):

        self.encoder = network.encoder.to(device)
        self.decoder = network.decoder.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        super().__init__(optimizer,lr,device)


    def forward(self, batch_data, cond=None):
        mu_e, log_var_e = self.encoder(batch_data)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        mu_d = self.network(batch_data)
        loss = self.loss_function(batch_data, mu_d)
        return loss

    def loss_function(self, x, mu_d, dim=-1):
        x_one_hot = F.one_hot(x.long(), num_classes=self.network.scatter_dim)
        log_p = x_one_hot * torch.log(torch.clamp(mu_d, EPS, 1. - EPS))
        return -torch.mean(log_p, dim).sum(-1).mean()

    def train(self, batch_data, cond=None):
        self.optimizer.zero_grad()
        loss = self.forward(batch_data, cond=cond)
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





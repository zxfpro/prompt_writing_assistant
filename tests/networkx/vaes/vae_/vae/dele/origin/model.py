from vae.dele.vaebase import VAEModelBase,VAENetworkBase
import torch

class Model(VAEModelBase):
    name = 'vae_origin11'
    def __init__(self,network:VAENetworkBase,lr,device='cpu'):

        self.encoder = network.encoder.to(device)
        self.decoder = network.decoder.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        super().__init__(optimizer,lr,device)

    def forward(self, batch_data,cond=None):
        mu_e, log_var_e = self.encoder(batch_data)
        z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
        RE = self.decoder.log_prob(batch_data, z)
        KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)
        return -(RE + KL).mean()

    def train(self,batch_data,cond=None):
        self.optimizer.zero_grad()
        loss = self.forward(batch_data,cond=cond)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def generate(self, batch_size, cond=None):
        z = self.prior.sample(batch_size=batch_size)
        return self.decoder.sample(z)

    def get_information(self):
        return super().get_information() +""""""


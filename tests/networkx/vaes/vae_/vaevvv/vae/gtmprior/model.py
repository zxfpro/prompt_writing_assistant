import torch.optim as optim
from ..base import ModelBase,NetworkBase
from .process import Process

class Model(ModelBase):
    """
    .train
    .generate

    """
    def __init__(self, network:NetworkBase,
                 lr:float,
                 device='cpu',
                 distribution='categorical',
                 num_components = 4,
                 gtm_type = 'gtm'):
        super().__init__()
        self.process = Process(latent=network.latent_dim,D=network.D,num_vals=network.num_vals,
                               distribution=distribution,num_components=num_components,gtm_type=gtm_type)
        self.network = network.to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.device = device

    def train_epoch(self,batch_data,cond=None):
        self.optimizer.zero_grad()
        loss = self.process.loss_function(batch_data, self.network.encoder, self.network.decoder)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def lightning_train(self,batch_data,cond=None):
        loss = self.process.loss_function(batch_data, self.network.encoder, self.network.decoder)
        return loss

    def generate(self, batch_size, cond=None):
        return self.process.sample(self.network.decoder, batch_size=batch_size)




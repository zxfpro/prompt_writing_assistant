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
                 type='gtm_vamp'):
        super().__init__()

        self.process = Process(latent=network.latent_dim,num_vals=network.num_vals,D=network.D,distribution=distribution,
                               type=type)
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
        return self.process.sample(decode_model = self.network.decoder,
                                   encode_model = self.network.encoder,
                                   batch_size=batch_size)




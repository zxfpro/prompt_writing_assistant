from typing import *
import torch.nn as nn

from sdmetrics.reports.multi_table import QualityReport
Qualreport = QualityReport()
Qualreport.generate(

class VAENetworkBase(nn.Module):
    data_demand = None
    def __init__(self,input_size:Tuple,encoder_network:Callable[[],object],decoder_network:Callable[[],object],
                 latent_dim = 16
                 ):
        super().__init__()
        self.input_size = input_size
        self.encoder_network = encoder_network
        self.decoder_network = decoder_network
        self.latent_dim = latent_dim

    def get_information(self):
        return f"""
network info:
input_size      :{self.input_size}
encoder_network :{self.encoder_network}
decoder_network :{self.decoder_network}
latent_dim      :{self.latent_dim}
"""

    def __repr__(self):
        return self.get_information()

class VAEModelBase(nn.Module):
    name = None

    def __init__(self, optimizer, lr, device):
        """
        network:VAENetwork,
        lr,
        device='cpu'):
        """
        super().__init__()
        self.encoder_network = None
        self.decoder_network = None
        self.optimizer = optimizer
        self.lr = lr
        self.device = device

    def forward(self, batch_data, cond=None):
        ...

    def loss_function(self):
        ...

    def train(self, batch_data):
        raise NotImplemented

    def generate(self, batch_size: int, cond=None):
        raise NotImplemented

    def get_information(self):
        return f"""
model info:
name            :{self.name}
lr              :{self.lr}
device          :{self.device}
        """

    def __repr__(self):
        return self.get_information()


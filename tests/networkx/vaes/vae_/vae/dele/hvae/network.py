
import torch.nn as nn
from vae.dele.vaebase import VAENetworkBase
import torch
from typing import *


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class HierarchicalVAEnetwork(VAENetworkBase):
    data_demand = None

    def __init__(self,
                 input_size: Tuple,
                 latent_dim:int,
                 middle = 256,
                 num_vals=None):
        M = middle

        self.nn_r_1 = nn.Sequential(nn.Linear(input_size[-1], M), GELU(),
                                   nn.BatchNorm1d(M),
                               nn.Linear(M, M), nn.GELU()
                              )

        self.nn_r_2 = nn.Sequential(nn.Linear(M, M), GELU(),
                                   nn.BatchNorm1d(M),
                               nn.Linear(M, M), nn.LeakyReLU()
                              )

        self.nn_delta_1 = nn.Sequential(nn.Linear(M, M), GELU(),
                                   nn.BatchNorm1d(M),
                                   nn.Linear(M, 2 * (latent_dim * 2)),
                                   )

        self.nn_delta_2 = nn.Sequential(nn.Linear(M, M), GELU(),
                                   nn.BatchNorm1d(M),
                                   nn.Linear(M, 2 * latent_dim),
                                   )

        self.nn_z_1 = nn.Sequential(nn.Linear(latent_dim, M), GELU(),
                               nn.BatchNorm1d(M),
                               nn.Linear(M, 2 * (latent_dim * 2))
                              )

        self.nn_x = nn.Sequential(nn.Linear(latent_dim * 2, M), GELU(),
                             nn.BatchNorm1d(M),
                             nn.Linear(M,M), GELU(),
                             nn.BatchNorm1d(M),
                             nn.Linear(M, input_size[-1] * num_vals)
                            )
    def forward(self,batch):
        pass

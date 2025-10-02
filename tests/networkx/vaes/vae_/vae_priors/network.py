import torch
import torch.nn as nn
from typing import *
from ..base import NetworkBase
from ..nn_function import set_linear
from ..utils import check_nnseq

# from ..layertools import MLP, CausalConv, Unsqueeze, View
# import numpy as np
# import torch.nn as nn
# def set_linear(input_size,output_size,dim_channel_list,scatter_dim):
#     in_channel = np.prod(input_size)
#     out_channel = np.prod(output_size)
#     network = nn.Sequential(View(),
#                             MLP([in_channel, *dim_channel_list, scatter_dim * out_channel],
#                                 shape_len=len(input_size), active='relu'),
#                             View((*input_size, scatter_dim)),
#                             nn.Softmax(), )
#     return network
#
# def set_conv(input_size, kernel_size, latent_dim, cond_size):
#     if len(input_size) == 1:
#         network = nn.Sequential(
#             Unsqueeze(1),
#             CausalConv([1, 30, 20, 21], shape_len=len(input_size), active='leakyrelu', kernel_size=kernel_size)
#         )
#
#     return network

class Network(NetworkBase):
    data_demand = ['column', 'uncond', 'scatter']

    def __init__(self, input_size:Tuple,nnseq:Callable[[torch.Tensor],object] or None=None ,
                 dim_channel_list=[20, 40, 60],
                 scatter_dim=64):
        if nnseq is None:
            network_obj = nn.Sequential(
                set_linear(input_size = input_size,output_size = (*input_size, scatter_dim),
                                     dim_channel_list=dim_channel_list),
                nn.Softmax())
        else:
            network_obj = nn.Sequential(nnseq,nn.Softmax())
            check_nnseq(input_size=input_size,output_size=(*input_size, scatter_dim),nn_obj=nnseq)

        super().__init__(input_size=input_size,dim_channel_list=dim_channel_list,
                         network=network_obj)
        self.scatter_dim = scatter_dim

    def forward(self, batch):
        return self.network(batch)

    def get_information(self):
        return super().get_information() + f"""scatter_dim : {self.scatter_dim}
        """



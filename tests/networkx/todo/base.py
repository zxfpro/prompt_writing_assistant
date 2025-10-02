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

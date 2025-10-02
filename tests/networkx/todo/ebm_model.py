# import torch.nn as nn
# from typing import *
#
# class NetworkBase(nn.Module):
#     data_demand = None
#     def __init__(self,input_size:Tuple,network:Callable[[],object],dim_channel_list:List):
#         super().__init__()
#         self.input_size = input_size
#         self.network = network
#         self.dim_channel_list = dim_channel_list
#
#     def get_information(self):
#         return f"""
# network info:
# data_demand     :{self.data_demand}
# input_size      :{self.input_size}
# dim_channel_list :{self.dim_channel_list}
# """
#
#     def __repr__(self):
#         return self.get_information()
#
# class ModelBase(nn.Module):
#     name = None
#     def __init__(self,optimizer,lr,device):
#         """
#         network:VAENetwork,
#         lr,
#         device='cpu'):
#         """
#         super(ModelBase, self).__init__()
#         self.network = None
#         self.optimizer = optimizer
#         self.lr = lr
#         self.device = device
#
#     def forward(self,batch_data,cond=None):
#         ...
#     def loss_function(self):
#         ...
#
#     def train(self,batch_data):
#         raise NotImplemented
#
#     def generate(self,batch_size:int,cond=None):
#         raise NotImplemented
#
#     def get_information(self):
#         return f"""
# models info:
# name            :{self.name}
# lr              :{self.lr}
# device          :{self.device}
#         """ + self.network.get_information()
#
#     def __repr__(self):
#         return self.get_information()
#
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
#
# class Model(ModelBase):
#     name = 'EBM'
#     def __init__(self, network: NetworkBase, lr, ld_steps=20, alpha=1.0, sigma=0.01, device='cpu'):
#         optimizer = optim.Adam(network.parameters(), lr=lr),
#         super().__init__(optimizer = optimizer,lr=lr,device=device)
#         self.network = network.to(device)
#         self.nll = nn.NLLLoss(reduction='none')
#         self.ld_steps = ld_steps
#         self.alpha = torch.FloatTensor([alpha])
#         self.sigma = sigma
#
#     def forward(self, batch_data, cond=None):
#         y = self.network.network(batch_data)
#         return y
#
#     def loss_function(self, x, cond, y):
#         loss_clf = self.class_loss(y, cond)
#         loss_gen = self.gen_loss(x, y)
#         loss = (loss_clf + loss_gen).mean()
#         self.loss_clf = loss_clf.mean()
#         self.loss_gen = loss_gen.mean()
#         return loss
#
#     def class_loss(self, y, cond):
#         y_pred = torch.softmax(y, 1)
#         # cond = cond.reshape(-1)#TODO
#         # cond = cond.long()
#
#         return self.nll(torch.log(y_pred), cond)
#
#     def gen_loss(self, x, y):
#         x_sample = self.generate(batch_size=x.shape[0])
#         # - calculate f(x_sample)[cond]
#         y_sample = self.network.network(x_sample)
#         return -(torch.logsumexp(y, 1) - torch.logsumexp(y_sample, 1))
#
#     def train(self, batch_data, cond=None):
#         self.optimizer.zero_grad()
#         y = self.forward(batch_data, cond)
#         loss = self.loss_function(batch_data,cond,y)
#         loss.backward(retain_graph=True)
#         self.optimizer.step()
#         return loss.item(), self.loss_clf.item(), self.loss_gen.item()
#
#     def energy_gradient(self, x):
#         # dataset original data that doesn't require grads!
#
#         # x_i = torch.FloatTensor(x.data)
#         x_i = x.data
#         x_i.requires_grad = True  # WE MUST ADD IT, otherwise autograd won't syn_2
#         # calculate the gradient
#         x_i.type_as(x)
#         x_i_grad = torch.autograd.grad(torch.logsumexp(self.network.network(x_i), 1).sum(), [x_i], retain_graph=True)[0]
#         # self.train()
#         return x_i_grad
#
#     def langevin_dynamics_step(self, x_old, alpha):
#         # Calculate gradient wrt x_old
#         grad_energy = self.energy_gradient(x_old)
#         # Sample eta ~ Normal(0, alpha)
#         epsilon = torch.randn_like(grad_energy) * self.sigma
#         # New sample
#         alpha = alpha.type_as(x_old)
#         x_new = x_old + alpha * grad_energy + epsilon
#         return x_new
#
#     def generate(self, batch_size, cond=None):
#         # - 1) Sample from uniform
#         x_sample = 2. * torch.rand([batch_size, self.network.in_channel], device=self.device) - 1.
#         # - 2) run Langevine Dynamics
#         for i in range(self.ld_steps):
#             x_sample = self.langevin_dynamics_step(x_sample, alpha=self.alpha)
#         return x_sample
#
#     def get_information(self):
#         return super().get_information() + \
# f"""
# ld_steps:     {self.ld_steps}
# alpha:        {self.alpha}
# sigma:        {self.sigma}
# """
#
#
#
# from typing import *
# import torch
#
# class Network(NetworkBase):
#     data_demand = ['column', 'cond', 'scatter']
#
#     def __init__(self, input_size:Tuple, nnseq:Callable[[torch.Tensor],object] or None=None ,
#                  dim_channel_list:List=[30, 40, 60],
#                  scatter_dim=32):
#
#         if nnseq is None:
#             network_obj = set_linear(input_size = input_size,output_size = scatter_dim,
#                                      dim_channel_list=dim_channel_list)
#             # def set_linear(input_size, dim_channel_list, scatter_dim):
#             #     in_channel = np.prod(input_size)
#             #     network = nn.Sequential(View(),
#             #                             MLP([in_channel, *dim_channel_list, scatter_dim], shape_len=len(input_size),
#             #                                 active='elu'))
#             #     return network
#             # network = set_linear(input_size, dim_channel_list)
#         else:
#             network_obj = nnseq
#             check_nnseq(input_size=input_size,output_size=(*input_size, scatter_dim),nn_obj=network_obj)
#
#         super().__init__(input_size=input_size,
#                                      dim_channel_list=dim_channel_list,
#                                      network=network_obj)
#         self.scatter_dim = scatter_dim
#
#     def forward(self, batch):
#         return self.network(batch)
#
#     def get_information(self):
#         return super().get_information() + f"""scatter_dim :     {self.scatter_dim}
# """
#
#
#
# import torch.nn as nn
# import numpy as np
# from gmodel.base import NetworkBase
# from gmodel.layertools import MLP,View
# from typing import *
#
#
# def set_linear(input_size, dim_channel_list,scatter_dim):
#     in_channel = np.prod(input_size)
#     network = nn.Sequential(View(),
#                             MLP([in_channel, *dim_channel_list, scatter_dim], shape_len=len(input_size),
#                                 active='elu'))
#     return network
#
# def set_conv(input_size, kernel_size, latent_dim, cond_size):
#     network = None
#     return network
#
#
# class EBMnet(NetworkBase):
#     data_demand = ['column', 'cond', 'scatter']
#
#     def __init__(self, input_size:Tuple, dim_channel_list:List=[30, 40, 60], scatter_dim=32):
#         network = set_linear(input_size, dim_channel_list)
#
#         super(EBMnet, self).__init__(input_size=input_size,network=network,
#                                      dim_channel_list=dim_channel_list)
#         # scatter_dim 和数据集有关
#         self.scatter_dim = scatter_dim
#
#     def forward(self, batch):
#         return self.network(batch)
#
#     def get_information(self):
#         return super(EBMnet, self).get_information() + f"""scatter_dim :     {self.scatter_dim}
# """
#
# import ebmodel
#
# process = ebmodel.origin.EBMProcess()
#
# network = ebmodel.origin.Network(input_size=(10,20),scatter_dim=5)
#
# models = ebmodel.origin.Model(network,lr=1e-3,process=process)
#
# import torch
# a = torch.randn(5,10,20)
#
# b = torch.ones(5)
#
# models.train(a,cond = b
#
# ## myself
# import torch
#
#
# class RBM():
#
#     def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4,
#                  use_cuda=True):
#         self.num_visible = num_visible
#         self.num_hidden = num_hidden
#         self.k = k
#         self.learning_rate = learning_rate
#         self.momentum_coefficient = momentum_coefficient
#         self.weight_decay = weight_decay
#         self.use_cuda = use_cuda
#
#         self.weights = torch.randn(num_visible, num_hidden) * 0.1
#         self.visible_bias = torch.ones(num_visible) * 0.5
#         self.hidden_bias = torch.zeros(num_hidden)
#
#         self.weights_momentum = torch.zeros(num_visible, num_hidden)
#         self.visible_bias_momentum = torch.zeros(num_visible)
#         self.hidden_bias_momentum = torch.zeros(num_hidden)
#
#         if self.use_cuda:
#             self.weights = self.weights.cuda()
#             self.visible_bias = self.visible_bias.cuda()
#             self.hidden_bias = self.hidden_bias.cuda()
#
#             self.weights_momentum = self.weights_momentum.cuda()
#             self.visible_bias_momentum = self.visible_bias_momentum.cuda()
#             self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()
#
#     def sample_hidden(self, visible_probabilities):
#         hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
#         hidden_probabilities = self._sigmoid(hidden_activations)
#         return hidden_probabilities
#
#     def sample_visible(self, hidden_probabilities):
#         visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
#         visible_probabilities = self._sigmoid(visible_activations)
#         return visible_probabilities
#
#     def gibbs(self, visible_probabilities):
#         hidden_probabilities = self.sample_hidden(visible_probabilities)
#         hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
#         visible_probabilities = self.sample_visible(hidden_activations)
#         return visible_probabilities
#
#     def contrastive_divergence(self, input_data):
#         # Positive phase
#         positive_hidden_probabilities = self.sample_hidden(input_data)
#         positive_hidden_activations = (
#                     positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
#         positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)
#
#         # Negative phase
#         hidden_activations = positive_hidden_activations
#
#         for step in range(self.k):
#             visible_probabilities = self.sample_visible(hidden_activations)
#             hidden_probabilities = self.sample_hidden(visible_probabilities)
#             hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
#
#         negative_visible_probabilities = visible_probabilities
#         negative_hidden_probabilities = hidden_probabilities
#
#         negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)
#
#         # Update parameters
#         self.weights_momentum *= self.momentum_coefficient
#         self.weights_momentum += (positive_associations - negative_associations)
#
#         self.visible_bias_momentum *= self.momentum_coefficient
#         self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)
#
#         self.hidden_bias_momentum *= self.momentum_coefficient
#         self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)
#
#         batch_size = input_data.size(0)
#
#         self.weights += self.weights_momentum * self.learning_rate / batch_size
#         self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
#         self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size
#
#         self.weights -= self.weights * self.weight_decay  # L2 weight decay
#
#         # Compute reconstruction error
#         error = torch.sum((input_data - negative_visible_probabilities) ** 2)
#
#         return error
#
#     def _sigmoid(self, x):
#         return 1 / (1 + torch.exp(-x))
#
#     def _random_probabilities(self, num):
#         random_probabilities = torch.rand(num)
#
#         if self.use_cuda:
#             random_probabilities = random_probabilities.cuda()
#
#         return random_probabilities
#
#
# CUDA = torch.cuda.is_available()
# CUDA_DEVICE = 1
# if CUDA:
#     torch.cuda.set_device(CUDA_DEVICE)
#
# sigmoid = torch.nn.Sigmoid()
# ########## LOADING DATASET ##########
# print('Loading dataset...')
#
# dataset = dev.Table(pd.DataFrame(data))
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
#
# ########## TRAINING RBM ##########
# print('Training RBM...')
#
# rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)
#
# for epoch in range(EPOCHS):
#     epoch_error = 0.0
#
#     for batch, _ in train_loader:
#         batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
#         if CUDA:
#             batch = batch.cuda()
#
#         batch_error = rbm.contrastive_divergence(batch)
#         epoch_error += batch_error
#
#     if epoch % 100 == 0:
#         print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))
#
#
#
#
#
#
#
#
#
#
#

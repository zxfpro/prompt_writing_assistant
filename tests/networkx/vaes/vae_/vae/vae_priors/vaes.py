# from gmodel.vae_priors.priors import *
# from gmodel.base import ModelBase,NetworkBase
#
#
# # network
# class Encoder(nn.Module):
#     def __init__(self,encoder):
#         super().__init__()
#         self.encoder = encoder
#
#     def reparameterization(self, mu, log_var):
#         std = torch.exp(0.5*log_var)
#         eps = torch.randn_like(std)
#         return mu + std * eps
#
#     def encode(self, x):
#         h_e = self.encoder(x)
#         mu_e, log_var_e = torch.chunk(h_e, 2, dim=1)
#         return mu_e, log_var_e
#
#     def sample(self, x=None, mu_e=None, log_var_e=None):
#         if (mu_e is None) and (log_var_e is None):
#             mu_e, log_var_e = self.encode(x)
#         else:
#             if (mu_e is None) or (log_var_e is None):
#                 raise ValueError('mu and log-scale can`t be None!')
#         z = self.reparameterization(mu_e, log_var_e)
#         return z
#
#     def log_prob(self, x=None, mu_e=None, log_var_e=None, z=None):
#         if x is not None:
#             mu_e, log_var_e = self.encode(x)
#             z = self.sample(mu_e=mu_e, log_var_e=log_var_e)
#         else:
#             if (mu_e is None) or (log_var_e is None) or (z is None):
#                 raise ValueError('mu, log-scale and z can`t be None!')
#
#         return log_normal_diag(z, mu_e, log_var_e)
#
#     def forward(self, x, type='log_prob'):
#         assert type in ['encode', 'log_prob'], 'Type could be either encode or log_prob'
#         if type == 'log_prob':
#             return self.log_prob(x)
#         else:
#             return self.sample(x)
# class Decoder(nn.Module):
#     def __init__(self, decoder_net, distribution='categorical', num_vals=None):
#         super(Decoder, self).__init__()
#
#         self.decoder = decoder_net
#         self.distribution = distribution
#         self.num_vals=num_vals
#
#     def decode(self, z):
#         h_d = self.decoder(z)
#         if self.distribution == 'categorical':
#             b = h_d.shape[0]
#             d = h_d.shape[1]//self.num_vals
#             h_d = h_d.view(b, d, self.num_vals)
#             mu_d = torch.softmax(h_d, 2)
#             return [mu_d]
#
#         elif self.distribution == 'bernoulli':
#             mu_d = torch.sigmoid(h_d)
#             return [mu_d]
#
#         else:
#             raise ValueError('Either `categorical` or `bernoulli`')
#     def sample(self, z):
#         outs = self.decode(z)
#
#         if self.distribution == 'categorical':
#             mu_d = outs[0]
#             b = mu_d.shape[0]
#             m = mu_d.shape[1]
#             mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)
#             p = mu_d.view(-1, self.num_vals)
#             x_new = torch.multinomial(p, num_samples=1).view(b, m)
#
#         elif self.distribution == 'bernoulli':
#             mu_d = outs[0]
#             x_new = torch.bernoulli(mu_d)
#
#         else:
#             raise ValueError('Either `categorical` or `bernoulli`')
#
#         return x_new
#
#     def log_prob(self, x, z):
#         outs = self.decode(z)
#         if self.distribution == 'categorical':
#             mu_d = outs[0]
#             log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)
#
#         elif self.distribution == 'bernoulli':
#             mu_d = outs[0]
#             log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)
#
#         else:
#             raise ValueError('Either `categorical` or `bernoulli`')
#         return log_p
#
#     def forward(self, z, x=None, type='log_prob'):
#         assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
#         if type == 'log_prob':
#             return self.log_prob(x, z)
#         else:
#             return self.sample(x)
#
#
# class VAENetwork(NetworkBase):
#     data_demand = None
#     def __init__(self,input_size,L=16,num_vals=64):
#         """
#         input_size  : window_size D
#         L 隐空间的维度   原始为16    更新为2
#         num——vals  ： digit——dim  分类维数
#         """
#         super(VAENetwork, self).__init__()
#         self.input_size = input_size
#         self.encoder_net,self.decoder_net = self.set_linear(window_size=input_size[-1],L=L,num_vals=num_vals)
#         self.num_vals = num_vals
#         self.L = L
#
#     def set_linear(self,window_size,num_vals,L = 2,M = 256):
#         """
#         D = window_size
#         """
#
#         encoder_net = nn.Sequential(nn.Linear(window_size, M), nn.LeakyReLU(),
#                                     nn.Linear(M, M), nn.LeakyReLU(),
#                                     nn.Linear(M, 2 * L))
#
#         decoder_net = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),
#                                     nn.Linear(M, M), nn.LeakyReLU(),
#                                     nn.Linear(M, num_vals * window_size))
#         return encoder_net,decoder_net
#
#     def forward(self,batch):
#         pass
#
#     def __repr__(self):
#         return f"""
# data_demand     :{self.data_demand}
# input_size      :{self.input_size}
# num_vals       :{self.num_vals}
# L              :{self.L}
# """
#
#
# class VAEModel(ModelBase):
#     name = 'VAEpro'
#     def __init__(self, network:VAENetwork,lr,digit_dim=64,prior_name='vampprior', z_dim=16, likelihood_type='categorical',device='cpu'):
#         super(VAEModel, self).__init__()
#         encoder = network.encoder_net
#         decoder = network.decoder_net
#         self.encoder = Encoder(encoder)
#         self.decoder = Decoder(decoder,distribution = likelihood_type,num_vals=digit_dim)
#
#         if prior_name in ['standard', 'flow2']:
#             num_components = 1
#         elif prior_name[0:3] == 'gtm':
#             num_components = 4
#         else:
#             num_components = 4 ** 2
#
#         self.L = network.L
#         self.D = network.input_size[-1]
#         self.num_vals = network.num_vals
#         self.prior_name = prior_name
#         # Second, we initialize the prior
#
#         if prior_name  == 'origin':
#             self.prior = Prior(L=z_dim)
#         elif prior_name == 'vampprior':
#             self.prior = VampPrior(L=self.L, D=self.D, num_vals=self.num_vals, encoder=self.encoder, num_components=num_components)
#         elif prior_name == 'standard':
#             self.prior = StandardPrior(L=self.L)
#         elif prior_name == 'gtm':
#
#             gtm_net = nn.Sequential(nn.Linear(2, 256), nn.Tanh(),
#                                     nn.Linear(256, 256), nn.Tanh(),
#                                     nn.Linear(256, 2 * self.L))
#
#             self.prior = GTMPrior(L=self.L, gtm_net=gtm_net, num_components=num_components, u_min=-10., u_max=10.)
#         elif prior_name == 'gtm-vampprior':
#             gtm_net_vamp = nn.Sequential(nn.Linear(2, 256), nn.Tanh(),
#                                          nn.Linear(256, 256), nn.Tanh(),
#                                          nn.Linear(256, self.D), nn.Sigmoid())
#             self.prior = GTMVampPrior(L=self.L, D=self.D, gtm_net=gtm_net_vamp, encoder=encoder, num_points=num_components,
#                                  u_min=-10., u_max=10., num_vals=self.num_vals)
#         elif prior_name == 'flow2':
#             num_flows = 3
#
#             # scale (s) network
#             nets = lambda: nn.Sequential(nn.Linear(self.L // 2, 256), nn.LeakyReLU(),
#                                          nn.Linear(256, 256), nn.LeakyReLU(),
#                                          nn.Linear(256, self.L // 2), nn.Tanh())
#
#             # translation (t) network
#             nett = lambda: nn.Sequential(nn.Linear(self.L // 2, 256), nn.LeakyReLU(),
#                                          nn.Linear(256, 256), nn.LeakyReLU(),
#                                          nn.Linear(256, self.L // 2))
#
#             self.prior = FlowPrior(nets, nett, num_flows=num_flows, D=self.L)
#         self.digit_dim = digit_dim
#         self.optimizer = torch.optim.Adamax([p for p in self.parameters() if p.requires_grad == True], lr=lr)
#         self.lr = lr
#         self.device = device
#
#
#     def forward(self,batch_data,reduction='avg'):
#         X,Y = batch_data
#         X = X.type(torch.FloatTensor)
#         mu_e, log_var_e = self.encoder.encode(X)
#         z = self.encoder.sample(mu_e=mu_e, log_var_e=log_var_e)
#
#         RE = self.decoder.log_prob(X, z)
#         KL = (self.prior.log_prob(z) - self.encoder.log_prob(mu_e=mu_e, log_var_e=log_var_e, z=z)).sum(-1)
#
#         error = 0
#         if np.isnan(RE.detach().numpy()).any():
#             print('RE {}'.format(RE))
#             error = 1
#         if np.isnan(KL.detach().numpy()).any():
#             print('RE {}'.format(KL))
#             error = 1
#
#         if error == 1:
#             raise ValueError()
#
#         if reduction == 'sum':
#             return -(RE + KL).sum()
#         else:
#             return -(RE + KL).mean()
#
#
#     def train(self, batch_data):
#         loss = self.forward(batch_data)
#         self.optimizer.zero_grad()
#         loss.backward(retain_graph=True)
#         self.optimizer.step()
#         return loss.item()
#
#     def generate(self, batch_size, cond=None):
#         z = self.prior.sample(batch_size=batch_size)
#         return self.decoder.sample(z)
#
#     def __repr__(self):
#         return f"""
# name            :{self.name}
# prior_name      :{self.prior_name}
# digit_dim       :{self.digit_dim}
# lr              :{self.lr}
# device          :{self.device}
#         """
#
#
#

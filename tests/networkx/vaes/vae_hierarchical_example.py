#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# **DISCLAIMER**
# 
# The presented code is not optimized, it serves an educational purpose. It is written for CPU, it uses only fully-connected networks and an extremely simplistic dataset. However, it contains all components that can help to understand how a hierarchical Variational Auto-Encoder (VAE) works, and it should be rather easy to extend it to more sophisticated models. This code could be run almost on any laptop/PC, and it takes a couple of minutes top to get the result.

# ## Dataset: Digits

# In this example, we go wild and use a dataset that is simpler than MNIST! We use a scipy dataset called Digits. It consists of ~1500 images of size 8x8, and each pixel can take values in $\{0, 1, \ldots, 16\}$.
# 
# The goal of using this dataset is that everyone can run it on a laptop, without any gpu etc.

# In[ ]:


class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
        else:
            self.data = digits.data[1350:].astype(np.float32)

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transforms:
            sample = self.transforms(sample)
        return sample


# ## Auxiliary functions and classes

# In[ ]:


# Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


# ### Distributions

# In[ ]:


PI = torch.from_numpy(np.asarray(np.pi))
EPS = 1.e-7

def log_categorical(x, p, num_classes=256, reduction=None, dim=None):
    x_one_hot = F.one_hot(x.long(), num_classes=num_classes)
    log_p = x_one_hot * torch.log(torch.clamp(p, EPS, 1. - EPS))
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_bernoulli(x, p, reduction=None, dim=None):
    pp = torch.clamp(p, EPS, 1. - EPS)
    log_p = x * torch.log(pp) + (1. - x) * torch.log(1. - pp)
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_normal_diag(x, mu, log_var, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

def log_standard_normal(x, reduction=None, dim=None):
    log_p = -0.5 * torch.log(2. * PI) - 0.5 * x**2.
    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p

# Chakraborty & Chakravarty, "A new discrete probability distribution with integer support on (−∞, ∞)",
#  Communications in Statistics - Theory and Methods, 45:2, 492-505, DOI: 10.1080/03610926.2013.830743

def log_min_exp(a, b, epsilon=1e-8):
    """
    Source: https://github.com/jornpeters/integer_discrete_flows
    Computes the log of exp(a) - exp(b) in a (more) numerically stable fashion.
    Using:
    log(exp(a) - exp(b))
    c + log(exp(a-c) - exp(b-c))
    a + log(1 - exp(b-a))
    And note that we assume b < a always.
    """
    y = a + torch.log(1 - torch.exp(b - a) + epsilon)

    return y

def log_integer_probability(x, mean, logscale):
    scale = torch.exp(logscale)

    logp = log_min_exp(
      F.logsigmoid((x + 0.5 - mean) / scale),
      F.logsigmoid((x - 0.5 - mean) / scale))

    return logp

def log_integer_probability_standard(x):
    logp = log_min_exp(
      F.logsigmoid(x + 0.5),
      F.logsigmoid(x - 0.5))

    return logp


# ## Hierarchical Variational Auto-Encoder

# **NOTE** Please note that we use a single class unlike in previous implementations of VAEs. We do it for clarity.

# In[ ]:


class HierarchicalVAE(nn.Module):
    def __init__(self, nn_r_1, nn_r_2, nn_delta_1, nn_delta_2, nn_z_1, nn_x, num_vals=256, D=64, L=16, likelihood_type='categorical'):
        super(HierarchicalVAE, self).__init__()

        print('Hierachical VAE by JT.')
        
        # bottom-up path
        self.nn_r_1 = nn_r_1
        self.nn_r_2 = nn_r_2
        
        self.nn_delta_1 = nn_delta_1
        self.nn_delta_2 = nn_delta_2
        
        # top-down path
        self.nn_z_1 = nn_z_1
        self.nn_x = nn_x

        
        # other params
        self.D = D
        
        self.L = L
        
        self.num_vals = num_vals
        
        self.likelihood_type = likelihood_type

    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
        
    def forward(self, x, reduction='avg'):
        #=====
        # bottom-up
        # step 1
        r_1 = self.nn_r_1(x)
        r_2 = self.nn_r_2(r_1)
        
        #step 2
        delta_1 = self.nn_delta_1(r_1)
        delta_mu_1, delta_log_var_1 = torch.chunk(delta_1, 2, dim=1)
        delta_log_var_1 = F.hardtanh(delta_log_var_1, -7., 2.)
        
        # step 3
        delta_2 = self.nn_delta_2(r_2)
        delta_mu_2, delta_log_var_2 = torch.chunk(delta_2, 2, dim=1)
        delta_log_var_2 = F.hardtanh(delta_log_var_2, -7., 2.)
        
        # top-down
        # step 4
        z_2 = self.reparameterization(delta_mu_2, delta_log_var_2)
        
        # step 5
        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        
        # step 6
        z_1 = self.reparameterization(mu_1 + delta_mu_1, log_var_1 + delta_log_var_1)
        
        # step 7
        h_d = self.nn_x(z_1)

        if self.likelihood_type == 'categorical':
            b = h_d.shape[0]
            d = h_d.shape[1]//self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)

        elif self.likelihood_type == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
        
        #=====ELBO
        # RE
        if self.likelihood_type == 'categorical':
            RE = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.likelihood_type == 'bernoulli':
            RE = log_bernoulli(x, mu_d, reduction='sum', dim=-1)
        
        # KL
        KL_z_2 = 0.5 * (delta_mu_2**2 + torch.exp(delta_log_var_2) - delta_log_var_2 - 1).sum(-1)
        KL_z_1 = 0.5 * (delta_mu_1**2 / torch.exp(log_var_1) + torch.exp(delta_log_var_1) -\
                        delta_log_var_1 - 1).sum(-1)
        
        KL = KL_z_1 + KL_z_2
        
        error = 0
        if np.isnan(RE.detach().numpy()).any():
            print('RE {}'.format(RE))
            print('KL {}'.format(KL))
            error = 1
        if np.isnan(KL.detach().numpy()).any():
            print('RE {}'.format(RE))            
            print('KL {}'.format(KL))
            error = 1

        if error == 1:
            raise ValueError()
        
        # Final ELBO
        if reduction == 'sum':
            loss = -(RE - KL).sum()
        else:
            loss = -(RE - KL).mean()
        
        return loss

    def sample(self, batch_size=64):
        # step 1
        z_2 = torch.randn(batch_size, self.L)
        # step 2
        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        # step 3
        z_1 = self.reparameterization(mu_1, log_var_1)
        
        # step 4
        h_d = self.nn_x(z_1)
        
        if self.likelihood_type == 'categorical':
            b = batch_size
            d = h_d.shape[1]//self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)
            # step 5
            p = mu_d.view(-1, self.num_vals)
            x_new = torch.multinomial(p, num_samples=1).view(b, d)

        elif self.likelihood_type == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
            # step 5
            x_new = torch.bernoulli(mu_d)
        return x_new


# ## Evaluation and Training functions

# **Evaluation step, sampling and curve plotting**

# In[ ]:


def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing model
        model_best = torch.load(name + '.model')

    model_best.eval()
    loss = 0.
    N = 0.
    for indx_batch, test_batch in enumerate(test_loader):
        loss_t = model_best.forward(test_batch, reduction='sum')
        loss = loss + loss_t.item()
        N = N + test_batch.shape[0]
    loss = loss / N

    if epoch is None:
        print(f'FINAL LOSS: nll={loss}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}')

    return loss


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x = next(iter(test_loader)).detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()


def samples_generated(name, data_loader, extra_name=''):
    x = next(iter(data_loader)).detach().numpy()

    # GENERATIONS-------
    model_best = torch.load(name + '.model')
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(batch_size=num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()


def plot_curve(name, nll_val):
    plt.plot(np.arange(len(nll_val)), nll_val, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + '_nll_val_curve.pdf', bbox_inches='tight')
    plt.close()


# **Training step**

# In[ ]:


def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, batch in enumerate(training_loader):
            loss = model.forward(batch)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_val = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_val)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.model')
            best_nll = loss_val
        else:
            if loss_val < best_nll:
                print('saved!')
                torch.save(model, name + '.model')
                best_nll = loss_val
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)

    return nll_val


# ## Experiments

# **Initialize datasets**

# In[ ]:


transforms = None


# In[ ]:


train_data = Digits(mode='train', transforms=transforms)
val_data = Digits(mode='val', transforms=transforms)
test_data = Digits(mode='test', transforms=transforms)

training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# **Hyperparameters**

# In[ ]:


D = 64   # input dimension
L = 8    # number of latents
M = 256  # the number of neurons in scale (s) and translation (t) nets

lr = 1e-3 # learning rate
num_epochs = 1000 # max. number of epochs
max_patience = 20 # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped


# **Creating a folder for results**

# In[ ]:


name = 'vae_hierarchical' + '_' + str(L)
result_dir ='results/' + name + '/'
if not(os.path.exists(result_dir)):
    os.mkdir(result_dir)


# **Initializing the model: (i) determining the conditional likelihood distribution, (ii) defininig encoder and decoder nets, and a prior**

# In[ ]:


likelihood_type = 'categorical'

if likelihood_type == 'categorical':
    num_vals = 17
elif likelihood_type == 'bernoulli':
    num_vals = 1


nn_r_1 = nn.Sequential(nn.Linear(D, M), GELU(),
                           nn.BatchNorm1d(M),
                       nn.Linear(M, M), nn.GELU()
                      )

nn_r_2 = nn.Sequential(nn.Linear(M, M), GELU(),
                           nn.BatchNorm1d(M),
                       nn.Linear(M, M), nn.LeakyReLU()
                      )

nn_delta_1 = nn.Sequential(nn.Linear(M, M), GELU(),
                           nn.BatchNorm1d(M),
                           nn.Linear(M, 2 * (L * 2)),
                           )

nn_delta_2 = nn.Sequential(nn.Linear(M, M), GELU(),
                           nn.BatchNorm1d(M),
                           nn.Linear(M, 2 * L),
                           )

nn_z_1 = nn.Sequential(nn.Linear(L, M), GELU(),
                       nn.BatchNorm1d(M),
                       nn.Linear(M, 2 * (L * 2))
                      )

nn_x = nn.Sequential(nn.Linear(L * 2, M), GELU(),
                     nn.BatchNorm1d(M),
                     nn.Linear(M,M), GELU(),
                     nn.BatchNorm1d(M),
                     nn.Linear(M, D * num_vals)
                    )


# Eventually, we initialize the full model
model = HierarchicalVAE(nn_r_1, nn_r_2, nn_delta_1, nn_delta_2, nn_z_1, nn_x, num_vals=num_vals, D=D, L=L, likelihood_type=likelihood_type)


# **Optimizer - here we use Adamax**

# In[ ]:


# OPTIMIZER
optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)


# **Training loop**

# In[ ]:


# Training procedure
nll_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs, model=model, optimizer=optimizer,
                       training_loader=training_loader, val_loader=val_loader)


# **The final evaluation**

# In[ ]:


test_loss = evaluation(name=result_dir + name, test_loader=test_loader)
f = open(result_dir + name + '_test_loss.txt', "w")
f.write(str(test_loss))
f.close()

samples_real(result_dir + name, test_loader)

plot_curve(result_dir + name, nll_val)

samples_generated(result_dir + name, test_loader, extra_name='FINAL')


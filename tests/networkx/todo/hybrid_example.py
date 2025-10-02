import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits
from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from pytorch_model_summary import summary


class Digits(Dataset):
    """Scikit-Learn Digits dataset."""

    def __init__(self, mode='train', transforms=None):
        digits = load_digits()
        if mode == 'train':
            self.data = digits.data[:1000].astype(np.float32)
            self.targets = digits.target[:1000]
        elif mode == 'val':
            self.data = digits.data[1000:1350].astype(np.float32)
            self.targets = digits.target[1000:1350]
        else:
            self.data = digits.data[1350:].astype(np.float32)
            self.targets = digits.target[1350:]

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_x = self.data[idx]
        sample_y = self.targets[idx]
        if self.transforms:
            sample_x = self.transforms(sample_x)
        return (sample_x, sample_y)

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


# Source: https://github.com/jornpeters/integer_discrete_flows
class RoundStraightThrough(torch.autograd.Function):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(ctx, input):
        rounded = torch.round(input, out=None)
        return rounded

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class HybridIDF(nn.Module):
    def __init__(self, netts, classnet, num_flows, alpha=1., D=2):
        super(HybridIDF, self).__init__()

        print('HybridIDF by JT.')

        if len(netts) == 1:
            self.t = torch.nn.ModuleList([netts[0]() for _ in range(num_flows)])
            self.idf_git = 1
            self.beta = nn.Parameter(torch.zeros(len(self.t)))

        elif len(netts) == 4:
            self.t_a = torch.nn.ModuleList([netts[0]() for _ in range(num_flows)])
            self.t_b = torch.nn.ModuleList([netts[1]() for _ in range(num_flows)])
            self.t_c = torch.nn.ModuleList([netts[2]() for _ in range(num_flows)])
            self.t_d = torch.nn.ModuleList([netts[3]() for _ in range(num_flows)])
            self.idf_git = 4
            self.beta = nn.Parameter(torch.zeros(len(self.t_a)))

        else:
            raise ValueError('You can provide either 1 or 4 translation nets.')

        self.classnet = classnet

        self.num_flows = num_flows

        self.round = RoundStraightThrough.apply

        self.mean = nn.Parameter(torch.zeros(1, D))
        self.logscale = nn.Parameter(torch.ones(1, D))

        self.D = D

        self.alpha = alpha

        self.nll = nn.NLLLoss(reduction='none')  # it requires log-softmax as input!!

    def coupling(self, x, index, forward=True):

        if self.idf_git == 1:
            (xa, xb) = torch.chunk(x, 2, 1)

            if forward:
                yb = xb + self.beta[index] * self.round(self.t[index](xa))
            else:
                yb = xb - self.beta[index] * self.round(self.t[index](xa))

            return torch.cat((xa, yb), 1)

        elif self.idf_git == 4:
            (xa, xb, xc, xd) = torch.chunk(x, 4, 1)

            if forward:
                ya = xa + self.beta[index] * self.round(self.t_a[index](torch.cat((xb, xc, xd), 1)))
                yb = xb + self.beta[index] * self.round(self.t_b[index](torch.cat((ya, xc, xd), 1)))
                yc = xc + self.beta[index] * self.round(self.t_c[index](torch.cat((ya, yb, xd), 1)))
                yd = xd + self.beta[index] * self.round(self.t_d[index](torch.cat((ya, yb, yc), 1)))
            else:
                yd = xd - self.beta[index] * self.round(self.t_d[index](torch.cat((xa, xb, xc), 1)))
                yc = xc - self.beta[index] * self.round(self.t_c[index](torch.cat((xa, xb, yd), 1)))
                yb = xb - self.beta[index] * self.round(self.t_b[index](torch.cat((xa, yc, yd), 1)))
                ya = xa - self.beta[index] * self.round(self.t_a[index](torch.cat((yb, yc, yd), 1)))

            return torch.cat((ya, yb, yc, yd), 1)

    def permute(self, x):
        return x.flip(1)

    def f(self, x):
        z = x
        for i in range(self.num_flows):
            z = self.coupling(z, i, forward=True)
            z = self.permute(z)

        return z

    def f_inv(self, z):
        x = z
        for i in reversed(range(self.num_flows)):
            x = self.permute(x)
            x = self.coupling(x, i, forward=False)

        return x

    def classify(self, x):
        z = self.f(x)
        y_pred = self.classnet(z)  # output: probabilities (i.e., softmax)
        return torch.argmax(y_pred, dim=1)

    def class_loss(self, x, y):
        z = self.f(x)
        y_pred = self.classnet(z)  # output: probabilities (i.e., softmax)
        return self.nll(torch.log(y_pred), y)

    def sample(self, batchSize):
        # sample z:
        z = self.prior_sample(batchSize=batchSize, D=self.D)
        # x = f^-1(z)
        x = self.f_inv(z)
        return x.view(batchSize, 1, self.D)

    def log_prior(self, x):
        log_p = log_integer_probability(x, self.mean, self.logscale)
        return log_p.sum(1)

    def prior_sample(self, batchSize, D=2):
        # Sample from logistic
        y = torch.rand(batchSize, self.D)
        x = torch.exp(self.logscale) * torch.log(y / (1. - y)) + self.mean
        # And then round it to an integer.
        return torch.round(x)

    def forward(self, x, y, reduction='avg'):
        z = self.f(x)
        y_pred = self.classnet(z)  # output: probabilities (i.e., softmax)

        idf_loss = -self.log_prior(z)
        class_loss = self.nll(torch.log(y_pred), y)  # remember to use logarithm on top of softmax!

        if reduction == 'sum':
            return (class_loss + self.alpha * idf_loss).sum()
        else:
            return (class_loss + self.alpha * idf_loss).mean()


# uxiliary functions: training, evaluation, plotting
# It's rather self-explanatory, isn't it?

def evaluation(test_loader, name=None, model_best=None, epoch=None):
    # EVALUATION
    if model_best is None:
        # load best performing models
        model_best = torch.load(name + '.models')

    model_best.eval()
    loss = 0.
    loss_error = 0.
    loss_gen = 0.
    N = 0.
    for indx_batch, (test_batch, test_targets) in enumerate(test_loader):
        # hybrid loss
        loss_t = model_best.forward(test_batch, test_targets, reduction='sum')
        loss = loss + loss_t.item()
        # classification error
        y_pred = model_best.classify(test_batch)
        e = 1.*(y_pred == test_targets)
        loss_error = loss_error + (1. - e).sum().item()
        # generative nll
        loss_i = (loss_t.item() - model_best.class_loss(test_batch, test_targets).sum().item()) / model_best.alpha
        loss_gen = loss_gen + loss_i
        # the number of examples
        N = N + test_batch.shape[0]
    loss = loss / N
    loss_error = loss_error / N
    loss_gen = loss_gen / N

    if epoch is None:
        print(f'FINAL PERFORMANCE: nll={loss}, ce={loss_error}, gen_nll={loss_gen}')
    else:
        print(f'Epoch: {epoch}, val nll={loss}, val ce={loss_error}, val gen_nll={loss_gen}')

    return loss, loss_error, loss_gen


def samples_real(name, test_loader):
    # REAL-------
    num_x = 4
    num_y = 4
    x, _ = next(iter(test_loader))
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name+'_real_images.pdf', bbox_inches='tight')
    plt.close()


def samples_generated(name, data_loader, extra_name=''):
    x, _ = next(iter(data_loader))
    x = x.detach().numpy()

    # GENERATIONS-------
    model_best = torch.load(name + '.models')
    model_best.eval()

    num_x = 4
    num_y = 4
    x = model_best.sample(num_x * num_y)
    x = x.detach().numpy()

    fig, ax = plt.subplots(num_x, num_y)
    for i, ax in enumerate(ax.flatten()):
        plottable_image = np.reshape(x[i], (8, 8))
        ax.imshow(plottable_image, cmap='gray')
        ax.axis('off')

    plt.savefig(name + '_generated_images' + extra_name + '.pdf', bbox_inches='tight')
    plt.close()

def plot_curve(name, nll_val, file_name='_nll_val_curve.pdf', color='b-'):
    plt.plot(np.arange(len(nll_val)), nll_val, color, linewidth='3')
    plt.xlabel('epochs')
    plt.ylabel('nll')
    plt.savefig(name + file_name, bbox_inches='tight')
    plt.close()


def training(name, max_patience, num_epochs, model, optimizer, training_loader, val_loader):
    nll_val = []
    gen_val = []
    error_val = []
    best_nll = 1000.
    patience = 0

    # Main loop
    for e in range(num_epochs):
        # TRAINING
        model.train()
        for indx_batch, (batch, targets) in enumerate(training_loader):
            loss = model.forward(batch, targets)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        # Validation
        loss_e, error_e, gen_e = evaluation(val_loader, model_best=model, epoch=e)
        nll_val.append(loss_e)  # save for plotting
        gen_val.append(gen_e)  # save for plotting
        error_val.append(error_e)  # save for plotting

        if e == 0:
            print('saved!')
            torch.save(model, name + '.models')
            best_nll = loss_e
        else:
            if loss_e < best_nll:
                print('saved!')
                torch.save(model, name + '.models')
                best_nll = loss_e
                patience = 0

                samples_generated(name, val_loader, extra_name="_epoch_" + str(e))
            else:
                patience = patience + 1

        if patience > max_patience:
            break

    nll_val = np.asarray(nll_val)
    error_val = np.asarray(error_val)
    gen_val = np.asarray(gen_val)

    return nll_val, error_val, gen_val

if __name__ == '__main__':

    # Initialize dataloaders
    train_data = Digits(mode='train')
    val_data = Digits(mode='val')
    test_data = Digits(mode='test')

    training_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    result_dir = 'results/'
    if not(os.path.exists(result_dir)):
        os.mkdir(result_dir)
    name = 'hybrididf'

    D = 64  # input dimension
    M = 256  # the number of neurons in scale (s) and translation (t) nets
    K = 10  # the number of labels

    alpha = 1. / D

    lr = 1e-3  # learning rate
    num_epochs = 1000  # max. number of epochs
    max_patience = 20  # an early stopping is used, if training doesn't improve for longer than 20 epochs, it is stopped

    # The number of invertible transformations
    num_flows = 2

    # This variable defines whether we use:
    #   1 - the classic coupling layer proposed in (Hogeboom et al., 2019)
    #   4 - the general invertible transformation in (Tomczak, 2020) with 4 partitions
    idf_git = 1

    if idf_git == 1:
        nett = lambda: nn.Sequential(nn.Linear(D // 2, M), nn.LeakyReLU(),
                                     nn.Linear(M, M), nn.LeakyReLU(),
                                     nn.Linear(M, D // 2))
        netts = [nett]

    elif idf_git == 4:
        nett_a = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                       nn.Linear(M, M), nn.LeakyReLU(),
                                       nn.Linear(M, D // 4))

        nett_b = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                       nn.Linear(M, M), nn.LeakyReLU(),
                                       nn.Linear(M, D // 4))

        nett_c = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                       nn.Linear(M, M), nn.LeakyReLU(),
                                       nn.Linear(M, D // 4))

        nett_d = lambda: nn.Sequential(nn.Linear(3 * (D // 4), M), nn.LeakyReLU(),
                                       nn.Linear(M, M), nn.LeakyReLU(),
                                       nn.Linear(M, D // 4))

        netts = [nett_a, nett_b, nett_c, nett_d]

    classnet = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                             nn.Linear(M, M), nn.LeakyReLU(),
                             nn.Linear(M, K),
                             nn.Softmax(dim=1))

    # Init IDF
    model = HybridIDF(netts, classnet, num_flows, D=D, alpha=alpha)
    # Print the summary (like in Keras)
    print(summary(model, torch.zeros(1, 64), torch.tensor([0]), show_input=False, show_hierarchical=False))

    # OPTIMIZER
    optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad == True], lr=lr)
    # Training procedure
    nll_val, error_val, gen_val = training(name=result_dir + name, max_patience=max_patience, num_epochs=num_epochs,
                                           model=model, optimizer=optimizer,
                                           training_loader=training_loader, val_loader=val_loader)
    test_loss, test_error, test_gen = evaluation(name=result_dir + name, test_loader=test_loader)
    f = open(result_dir + name + '_test_loss.txt', "w")
    f.write('NLL: ' + str(test_loss) + '\nCA: ' + str(test_error) + '\nGEN NLL: ' + str(test_gen))
    f.close()

    samples_real(result_dir + name, test_loader)

    plot_curve(result_dir + name, nll_val)
    plot_curve(result_dir + name, error_val, file_name='_ca_val_curve.pdf', color='r-')
    plot_curve(result_dir + name, gen_val, file_name='_gen_val_curve.pdf', color='g-')



















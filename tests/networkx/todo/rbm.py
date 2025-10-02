## myself
import torch

import torch.nn as nn

from typing import *
class NetworkBase(nn.Module):
    data_demand = None
    def __init__(self,input_size:Tuple,network:Callable[[],object],dim_channel_list:List):
        super().__init__()
        self.input_size = input_size
        self.network = network
        self.dim_channel_list = dim_channel_list

    def get_information(self):
        return f"""
network info:
data_demand     :{self.data_demand}
input_size      :{self.input_size}
dim_channel_list :{self.dim_channel_list}
"""

    def __repr__(self):
        return self.get_information()

class ModelBase(nn.Module):
    name = None
    def __init__(self,optimizer,lr,device):
        """
        network:VAENetwork,
        lr,
        device='cpu'):
        """
        super(ModelBase, self).__init__()
        self.network = None
        self.optimizer = optimizer
        self.lr = lr
        self.device = device

    def forward(self,batch_data,cond=None):
        ...
    def loss_function(self):
        ...

    def train(self,batch_data):
        raise NotImplemented

    def generate(self,batch_size:int,cond=None):
        raise NotImplemented

    def get_information(self):
        return f"""
models info:
name            :{self.name}
lr              :{self.lr}
device          :{self.device}
        """ + self.network.get_information()


    def __repr__(self):
        return self.get_information()







class RBM():

    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3, momentum_coefficient=0.5, weight_decay=1e-4,
                 use_cuda=True):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.learning_rate = learning_rate
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        self.weights = torch.randn(num_visible, num_hidden) * 0.1
        self.visible_bias = torch.ones(num_visible) * 0.5
        self.hidden_bias = torch.zeros(num_hidden)

        self.weights_momentum = torch.zeros(num_visible, num_hidden)
        self.visible_bias_momentum = torch.zeros(num_visible)
        self.hidden_bias_momentum = torch.zeros(num_hidden)

        if self.use_cuda:
            self.weights = self.weights.cuda()
            self.visible_bias = self.visible_bias.cuda()
            self.hidden_bias = self.hidden_bias.cuda()

            self.weights_momentum = self.weights_momentum.cuda()
            self.visible_bias_momentum = self.visible_bias_momentum.cuda()
            self.hidden_bias_momentum = self.hidden_bias_momentum.cuda()

    def sample_hidden(self, visible_probabilities):
        hidden_activations = torch.matmul(visible_probabilities, self.weights) + self.hidden_bias
        hidden_probabilities = self._sigmoid(hidden_activations)
        return hidden_probabilities

    def sample_visible(self, hidden_probabilities):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()) + self.visible_bias
        visible_probabilities = self._sigmoid(visible_activations)
        return visible_probabilities

    def gibbs(self, visible_probabilities):
        hidden_probabilities = self.sample_hidden(visible_probabilities)
        hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        visible_probabilities = self.sample_visible(hidden_activations)
        return visible_probabilities

    def contrastive_divergence(self, input_data):
        # Positive phase
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (
                    positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations

        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters
        self.weights_momentum *= self.momentum_coefficient
        self.weights_momentum += (positive_associations - negative_associations)

        self.visible_bias_momentum *= self.momentum_coefficient
        self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

        self.hidden_bias_momentum *= self.momentum_coefficient
        self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size

        self.weights -= self.weights * self.weight_decay  # L2 weight decay

        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities) ** 2)

        return error

    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num)

        if self.use_cuda:
            random_probabilities = random_probabilities.cuda()

        return random_probabilities


CUDA = torch.cuda.is_available()
CUDA_DEVICE = 1
if CUDA:
    torch.cuda.set_device(CUDA_DEVICE)

sigmoid = torch.nn.Sigmoid()
########## LOADING DATASET ##########
print('Loading dataset...')

dataset = dev.Table(pd.DataFrame(data))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

########## TRAINING RBM ##########
print('Training RBM...')

rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS, CD_K, use_cuda=CUDA)

for epoch in range(EPOCHS):
    epoch_error = 0.0

    for batch, _ in train_loader:
        batch = batch.view(len(batch), VISIBLE_UNITS)  # flatten input data
        if CUDA:
            batch = batch.cuda()

        batch_error = rbm.contrastive_divergence(batch)
        epoch_error += batch_error

    if epoch % 100 == 0:
        print('Epoch Error (epoch=%d): %.4f' % (epoch, epoch_error))


############

# RBM
import pandas as pd
from sklearn.neural_network import BernoulliRBM
RBM = BernoulliRBM




from sklearn.linear_model import LinearRegression,LogisticRegression,LogisticRegressionCV
from sklearn.linear_model import Lasso,LassoCV,Ridge,RidgeCV,RidgeClassifier
from sklearn.linear_model import MultiTaskLasso,MultiTaskLassoCV
from sklearn.svm import SVC,SVR,LinearSVC,LinearSVR

from sklearn import preprocessing

def labelencoder(col_1=["paris", "paris", "tokyo", "amsterdam"]):
    le = preprocessing.LabelEncoder()
    le.fit(col_1)
    return le.transform(col_1)
    # le.inverse_transform([1,2,2,1])



class sklearn_ML():
    """
            models__ = {  'LinearRegression':LinearRegression,
                    'LogisticRegression':LogisticRegression,
                    'Lasso':Lasso,
                    'Ridge':Ridge,
                    'RidgeClassifier':RidgeClassifier,
                    'MultiTaskLasso':MultiTaskLasso,
                    "LinearSVC":LinearSVC,
                     "LinearSVR":LinearSVR,
                     "SVC":SVC,
                     "SVR":SVR,
                    }

        models_CV = {
            'LogisticRegressionCV': LogisticRegressionCV,
            'LassoCV': LassoCV,
            'RidgeCV': RidgeCV,
            'MultiTaskLassoCV':MultiTaskLassoCV,
        }

    """
    def __init__(self,model:str,id:str,label:str):
        models = {  'LinearRegression':LinearRegression,
                    'LogisticRegression':LogisticRegression,
                    'Lasso':Lasso,
                    'Ridge':Ridge,
                    'RidgeClassifier':RidgeClassifier,
                    'MultiTaskLasso':MultiTaskLasso,
                    "LinearSVC":LinearSVC,
                     "LinearSVR":LinearSVR,
                     "SVC":SVC,
                     "SVR":SVR,
                    }

        models_CV = {
            'LogisticRegressionCV': LogisticRegressionCV,
            'LassoCV': LassoCV,
            'RidgeCV': RidgeCV,
            'MultiTaskLassoCV':MultiTaskLassoCV,
        }
        if model in models:
            print('in models')
            self.model = models[model]
        elif model in models_CV:
            print('in modelCV')
            self.model = models[model]
        self.id = id
        self.label = label

    def fit(self,traindata:pd.DataFrame):
        Xtrain = traindata.drop(columns=[self.id]).values
        Ytrain = traindata[self.label].values
        self.model.fit(Xtrain,Ytrain)

    def predict(self,testdata):

        if self.label in testdata.columns:
            preds = self.model.predict(testdata.drop(columns=[self.id,self.label]))
        else:
            preds = self.model.predict(testdata.drop(columns=[self.id]))
        return pd.DataFrame({self.id: testdata[self.id], self.label: preds})

    def accuary(self):
        pass





if __name__ == '__main__':
    rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=0, verbose=True)
    rbm.fit()


class Classification_Generation(RBM):
    # n_k is the number of classes
    def __init__(self, n_x, n_h, n_k, n_d=1, W=None, b=None, c=None, k=1, model_file=None):
        self.n_x = n_x
        self.n_k = n_k
        self.n_d = n_d
        # label is represented with n_k*n_d dots, where label i is the i-th group of n_d dots
        # np.zeros(n_k*n_d)[i*n_d:+n_d] = 1
        super().__init__(n_x + n_k * n_d, n_h, W, b, c, k)
        return

    def convert_input(self, X, Label):
        n_sample = X.shape[0]
        n_k = self.n_k;
        n_d = self.n_d;
        n_x = self.n_x
        assert X.shape == (n_sample, n_x)

        if Label is not None:
            Label = Label.flatten()
            # attach label to data
            Y = np.zeros(shape=(n_sample, n_k))
            Index = np.arange(n_sample)
            Y[Index, Label] = 1
            Y = Y.repeat(n_d, axis=1)
        else:
            Y = np.zeros(shape=(n_sample, n_k * n_d))

        V = np.append(X, Y, axis=1)
        assert V.shape == (n_sample, self.n_v)
        return V

    def train(self, X, Label, learning=0.01):
        V = self.convert_input(X, Label)
        self.contrastive_divergence(V, learning)
        return

    def classify(self, x):
        # set_trace()
        x = x.reshape(1, -1)
        v = self.convert_input(x, None)
        vp, vs = self.reconstruct(v)

        n_k = self.n_k;
        n_d = self.n_d;
        n_x = self.n_x
        result = vp[0, n_x:]
        result = result.reshape(n_k, -1).sum(axis=1)  # collapse the k*d dots into k sum dots.
        pred = np.argmax(result)

        return vp[0, :n_x], pred

    def generate(self, label, k_cd, init_w, w):
        # this is a very simple generation algorithm with single-layer RBM
        n_x = self.n_x;
        n_k = self.n_k;
        n_d = self.n_d

        classes = np.zeros(n_k)
        classes[label] = init_w
        classes = classes.repeat(n_d)

        np.random.seed(1234)
        #    v = np.append( np.random.binomial(1, 0.5, n_x), classes)
        #    v = np.append( np.zeros(n_x), classes)
        # v = np.append( np.random.uniform(0, 0.8, n_x), classes)
        v = np.append(np.full(n_x, 0.1), classes)
        vp = v.reshape(1, -1)

        classes = np.zeros(n_k)
        classes[label] = w
        classes = classes.repeat(n_d)

        for i in range(k_cd):
            vp, vs = self.reconstruct(vp)
            vp[0, n_x:] = classes

        return vp[0, :n_x]

    def save_model(self, save_file):
        dict = {'n_x': self.n_x, 'n_h': self.n_h, 'n_k': self.n_k, 'n_d': self.n_d, 'W': self.W, 'b': self.b,
                'c': self.c}
        save_file += ".(" + str(self.n_x) + "+" + str(self.n_k * self.n_d) + ")x" + str(self.n_h)
        with open(save_file, 'wb') as f:
            pickle.dump(dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        return

    @classmethod
    def load_model(cls, load_file):
        with open(load_file, 'rb') as f:
            m = pickle.load(f)

        rbm = cls(m['n_x'], m['n_h'], m['n_k'], m['n_d'], m['W'], m['b'], m['c'])
        return rbm


import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Grid
import math


class MNIST_RBM_CLASSIFY:
    def __init__(self, n_h, n_d=1, load_file=None, folder="../convolution-network"):
        self.rbm_done = False
        d = 28
        n_x = d * d;
        n_k = 10

        # set_trace()
        if load_file is None:
            self.rbm = Classification_Generation(n_x, n_h, n_k, n_d)
        else:
            self.rbm = Classification_Generation.load_model(load_file)
            self.rbm_done = True

        self.train_input = MnistInput("train", folder)
        self.test_input = MnistInput("test", folder)

        return

    def train(self, train_size=-1, n_epoch=100, batch_size=10, learning=0.1):
        if self.rbm_done: return

        n_ep = 0
        n_sample = 0
        X = []
        Y = []
        batch_size = batch_size if batch_size > 0 else 100
        n_epoch = n_epoch if n_epoch > 0 else 1
        startrate = learning
        for i in range(n_epoch):
            learning = startrate * math.pow(0.1, i // 50)
            for x, y in self.train_input.read(train_size):
                n_sample += 1
                X.append(x)
                Y.append(y)
                if n_sample >= batch_size:
                    X = np.array(X).reshape(batch_size, -1) > 30
                    X = X * 1  # make bool into number
                    self.rbm.train(X, np.array(Y), learning)
                    n_sample = 0
                    X = []
                    Y = []

        self.rbm.save_model("mnist_rbm_classify.epochs" + str(n_epoch))
        return

    def classify(self, test_size=-1, output_size=50):
        d = 28
        n_x = self.rbm.n_x
        n_k = self.rbm.n_k
        n_total = 0;
        n_correct = 0;
        n_output = 0
        X = [];
        Y = [];
        Recon = [];
        Preds = []
        for x, y in self.test_input.read(test_size):
            n_total += 1
            reco, pred = self.rbm.classify(x / 255)
            if y == pred: n_correct += 1
            if n_output < output_size:
                Preds.append(pred)
                Recon.append(reco)
                X.append(x)
                Y.append(y)
                n_output += 1

        accuracy = n_correct / n_total
        print("Accuracy: {}".format(accuracy))

        # output
        ncols = 10
        nrows = int(output_size / 5)
        fig = plt.figure(figsize=(ncols, int(nrows * 2)), dpi=100)
        grid = Grid(fig, rect=111, nrows_ncols=(nrows, ncols))

        for i, ax in enumerate(grid):
            j = i // 2
            if i % 2 == 0:
                ax.imshow(X[j].reshape(d, d), cmap=mpl.cm.Greys)
                ax.set_title("Orig: {}".format(Y[j]), y=0)
            else:
                ax.imshow(Recon[j].reshape(d, d), cmap=mpl.cm.Greys)
                ax.set_title("Reco: {}".format(Preds[j]), y=0)

            ax.set_axis_off()

        fig.suptitle('Original and reconstructed digits side by side')
        fig.tight_layout()
        fig.subplots_adjust(top=0.98)
        plt.show()
        return

    def generate(self, n_reco=50, init_w=30, w=30):
        # 50,30,30 is for mnist_rbm_classify.(784+10)x85.epochs100
        digits = []
        for i in range(self.rbm.n_k):
            digit = self.rbm.generate(i, n_reco, init_w, w)
            digits.append(digit)

        # output
        d = 28
        ncols = self.rbm.n_k
        nrows = 1
        fig = plt.figure(figsize=(ncols, int(nrows * 2)), dpi=100)
        grid = Grid(fig, rect=111, nrows_ncols=(nrows, ncols))

        for i, ax in enumerate(grid):
            ax.imshow(digits[i].reshape(d, d), cmap=mpl.cm.Greys)
            ax.set_title(i)
            ax.set_axis_off()

        fig.suptitle('Generated digit images from numbers')
        fig.tight_layout()
        fig.subplots_adjust(top=0.7)
        plt.show()
        return


mnist = None
if __name__ == "__main__" and '__file__' not in globals():
    np.seterr(all='raise')
    plt.close('all')
    mnist = MNIST_RBM_CLASSIFY(None, n_d=2, load_file="trained_models/mnist_rbm_classify.epochs10.(784+10)x85")
    #mnist = MNIST_RBM_CLASSIFY(n_h=28*3+1, n_d=2)
    mnist.train(train_size=-1, n_epoch=10, batch_size=10, learning=0.1)
    mnist.generate(20,50,8)  #mnist_rbm_classify.epochs10.(784+10)x85
    #mnist.classify()


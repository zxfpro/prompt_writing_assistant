
from gmodel.vae_priors.priors import *
from gmodel.base import NetworkBase
from gmodel.vae_priors.vaes import Decoder
from collections import OrderedDict
from gmodel.layertools import View
# class MLP(nn.Module):
#     def __init__(self, hidden_size, last_activation=True):
#         """
#         生成标准的全连接网络
#         hidden_size,全连接隐藏层的层数,list
#         last_activation,是否在全连接层最后加入BN层和ReLU层,BOOL
#         """
#         super().__init__()
#         q = []
#         for i in range(len(hidden_size) - 1):
#             in_dim = hidden_size[i]
#             out_dim = hidden_size[i + 1]
#             q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
#             if (i < len(hidden_size) - 2) or (
#                     (i == len(hidden_size) - 2) and (last_activation)
#             ):
#                 q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
#                 q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
#         self.mlp = nn.Sequential(OrderedDict(q))

#     def forward(self, x):
#         return self.mlp(x)

# class Reshape(nn.Module):
#     def __init__(self,*target_shape):
#         super().__init__()
#         self.target_shape = target_shape
#     def forward(self,x):
#         batch_size = x.shape[0]
#         return x.reshape((batch_size,)+self.target_shape)
class Conv1dTranspose_list(nn.Module):

    def __init__(self, hidden_layer_size):  # [200,100,4]
        super().__init__()
        net = OrderedDict()
        for i in range(len(hidden_layer_size) - 1):
            net[f'conv_trans{i}'] = nn.ConvTranspose1d(hidden_layer_size[i], hidden_layer_size[i + 1],
                                                       kernel_size=3, padding=1, output_padding=1, stride=2)
            net[f'relu_trans{i}'] = nn.ReLU()

        self.conv = nn.Sequential(net)

    def forward(self, x):
        # 128 200 63
        x = self.conv(x)
        # 128,4,252
        return x

class TimeDecoder(Decoder):
    def __init__(self, latent_dim, feat_dim, seq_len, hidden_layer_sizes, trend_poly=2, num_gen_seas=3,
                 use_residual_conn=True, use_scaler=True):
        super(TimeDecoder, self).__init__()
        # self.level = None
        # self.trend = None
        # self.decoder_residual = None
        # self.seasonal_freq = None
        # self.seasonal_phase = None
        # self.seasonal_amplitude = None
        self.latent_dim = latent_dim  # 8
        self.feat_dim = feat_dim  # 4
        self.seq_len = seq_len  # 252
        self.trend_poly = trend_poly
        self.num_gen_seas = num_gen_seas
        self.hidden_layer_sizes = hidden_layer_sizes  # [4,100,200]

        number_layer = 2 ** (len(self.hidden_layer_sizes) - 1)
        num_layer = int(self.seq_len / number_layer)

        self.encoder_last_dense_dim = self.hidden_layer_sizes[-1] * num_layer  # 200*63 = 12600
        self.use_residual_conn = use_residual_conn
        self.use_scaler = use_scaler

        self.level = nn.Sequential(
            nn.Linear(self.latent_dim, self.feat_dim),  # [128,4]
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),  # [128,4]
            View(self.feat_dim, 1)  # [128,4,1]
        )

        self.trend = nn.Sequential(
            nn.Linear(self.latent_dim, self.feat_dim * self.trend_poly),  # [128,8]
            nn.ReLU(),
            nn.Linear(self.feat_dim * self.trend_poly, self.feat_dim * self.trend_poly),
            View(self.feat_dim, self.trend_poly)  # [128,4,2]
        )

        self.decoder_residual = nn.Sequential(
            nn.Linear(self.latent_dim, self.encoder_last_dense_dim),
            # [128,200*63]
            nn.ReLU(),
            View(self.hidden_layer_sizes[-1], -1),  # [128,200,63]
            Conv1dTranspose_list(hidden_layer_size=list(reversed(self.hidden_layer_sizes))),
            # 128,4,252
            View(),
            nn.Linear(self.seq_len * self.feat_dim, self.seq_len * self.feat_dim),
            View(self.feat_dim, self.seq_len),
        )

        self.seasonal_freq = nn.Sequential(
            nn.Linear(self.latent_dim, self.feat_dim * self.num_gen_seas),
            nn.Sigmoid(),
            View(1, self.feat_dim, self.num_gen_seas),
        )

        self.seasonal_phase = nn.Sequential(
            nn.Linear(self.latent_dim, self.feat_dim * self.num_gen_seas),
            View(1, self.feat_dim, self.num_gen_seas),
        )

        self.seasonal_amplitude = nn.Sequential(
            nn.Linear(self.latent_dim, self.feat_dim * self.num_gen_seas),
            View(1, self.feat_dim, self.num_gen_seas),
        )

    def level_model(self, x):
        x = self.level(x)
        ones_tensor = torch.ones(1, 1, self.seq_len)  # 1,D,T
        ones_tensor = ones_tensor.type_as(x)
        return x * ones_tensor  # N D T
    def scale_model(self, x):
        x = self.level(x)  # 1,D,T
        return x.repeat(1, 1, self.seq_len)
    def trend_model(self, x):
        x = self.trend(x)
        lin_space = torch.arange(0, float(self.seq_len), 1) / self.seq_len
        poly_space = torch.stack([lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0)  # shape: P x T
        lin_space = lin_space.type_as(x)
        poly_space = poly_space.type_as(x)
        x = torch.matmul(x, poly_space)  # NDT
        x = x.to(torch.float32)  # N T D
        return x
    def _get_decoder_residual(self, x):
        x = self.decoder_residual(x)
        return x
    def generic_seasonal_model(self, x):
        freq = self.seasonal_freq(x)
        phase = self.seasonal_phase(x)
        amplitude = self.seasonal_amplitude(x)
        lin_space = torch.arange(0, float(self.seq_len),
                                 1) / self.seq_len  # shape of lin_space : 1d tensor of length T
        lin_space = lin_space.reshape(1, self.seq_len, 1, 1).type_as(x)

        seas_vals = amplitude * torch.sin(2. * np.pi * freq * lin_space + phase)  # shape: N, T, D, S
        seas_vals = torch.sum(seas_vals, axis=-1)
        seas_vals = seas_vals.permute(0, 2, 1)
        return seas_vals

    def decode(self,z):
        decoder_inputs = z
        outputs = None
        outputs = self.level_model(decoder_inputs)
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = self.trend_model(decoder_inputs)
            outputs = trend_vals if outputs is None else outputs + trend_vals

        if self.use_residual_conn:
            # [128,8]
            residuals = self._get_decoder_residual(decoder_inputs)
            # 128,4,252
            outputs = residuals if outputs is None else outputs + residuals

            # generic seasonalities
        if self.num_gen_seas is not None and self.num_gen_seas > 0:
            gen_seas_vals = self.generic_seasonal_model(decoder_inputs)
            outputs = gen_seas_vals if outputs is None else outputs + gen_seas_vals
            # custom seasons

        if self.use_scaler and outputs is not None:
            # [128,8]
            scale = self.scale_model(decoder_inputs)
            outputs *= scale
        outputs = outputs.permute(0, 2, 1)
        h_d = outputs
        if self.distribution == 'categorical':
            b = h_d.shape[0]
            d = h_d.shape[1]//self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)
            return [mu_d]

        elif self.distribution == 'bernoulli':
            mu_d = torch.sigmoid(h_d)
            return [mu_d]

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

    def sample(self, z):
        outs = self.decode(z)

        if self.distribution == 'categorical':
            mu_d = outs[0]
            b = mu_d.shape[0]
            m = mu_d.shape[1]
            mu_d = mu_d.view(mu_d.shape[0], -1, self.num_vals)
            p = mu_d.view(-1, self.num_vals)
            x_new = torch.multinomial(p, num_samples=1).view(b, m)

        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            x_new = torch.bernoulli(mu_d)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')

        return x_new

    def log_prob(self, x, z):
        outs = self.decode(z)
        if self.distribution == 'categorical':
            mu_d = outs[0]
            log_p = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.distribution == 'bernoulli':
            mu_d = outs[0]
            log_p = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        else:
            raise ValueError('Either `categorical` or `bernoulli`')
        return log_p

    def forward(self, z, x=None, type='log_prob'):
        assert type in ['decoder', 'log_prob'], 'Type could be either decode or log_prob'
        if type == 'log_prob':
            return self.log_prob(x, z)
        else:
            return self.sample(x)



class TimeVAENetwork(NetworkBase):
    data_demand = None

    def __init__(self, input_size, L=16, num_vals=17):
        super(TimeVAENetwork, self).__init__()
        # alpha = 3.0, trend_poly = 2, num_gen_seas = 3, use_residual_conn = True, use_scaler = True
        self.input_size = input_size
        self.encoder_net, self.decoder_net = self.set_linear(window_size=input_size[-1], L=L, num_vals=num_vals)
        self.num_vals = num_vals
        self.L = L

    def set_linear(self, window_size=64, L=2, M=256, num_vals=17):
        D = window_size

        encoder_net = nn.Sequential(nn.Linear(D, M), nn.LeakyReLU(),
                                    nn.Linear(M, M), nn.LeakyReLU(),
                                    nn.Linear(M, 2 * L))

        decoder_net = nn.Sequential(nn.Linear(L, M), nn.LeakyReLU(),
                                    nn.Linear(M, M), nn.LeakyReLU(),
                                    nn.Linear(M, num_vals * D))
        return encoder_net, decoder_net

    def __repr__(self):
        return f"""
data_demand     :{self.data_demand}
input_size      :{self.input_size}
num_vals       :{self.num_vals}
L              :{self.L}
"""


from gmodel.vae_priors.vaes import VAEModel


class TimeVAEModel(VAEModel):
    name = 'TimeVAE'

    def __init__(self, *wargs, **kwargs):
        super().__init__(*wargs, **kwargs)

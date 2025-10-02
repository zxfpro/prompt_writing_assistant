
import torch



class HierarchicalVAEmodel(ModelBase):
    name = 'HVAE'
    def __init__(self, network,lr,digit_dim=17, D=30, L=16,
                 likelihood_type='categorical',device='cpu'):
        super(HierarchicalVAEmodel, self).__init__()
        # D window
        # nn_r_1, nn_r_2, nn_delta_1, nn_delta_2, nn_z_1, nn_x,
        # bottom-up path
        self.nn_r_1 = network.nn_r_1.to(device)
        self.nn_r_2 = network.nn_r_2.to(device)

        self.nn_delta_1 = network.nn_delta_1.to(device)
        self.nn_delta_2 = network.nn_delta_2.to(device)

        # top-down path
        self.nn_z_1 = network.nn_z_1.to(device)
        self.nn_x = network.nn_x.to(device)

        # other params
        self.D = D
        self.L = L

        self.num_vals = digit_dim

        self.likelihood_type = likelihood_type
        self.optimizer = torch.optim.Adamax([p for p in self.parameters() if p.requires_grad == True], lr=lr)
        self.lr = lr
        self.device = device


    def reparameterization(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x,cond, reduction='avg'):
        # =====
        # bottom-up
        # step 1
        r_1 = self.nn_r_1(x)
        r_2 = self.nn_r_2(r_1)

        # step 2
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
            d = h_d.shape[1] // self.num_vals
            h_d = h_d.view(b, d, self.num_vals)
            mu_d = torch.softmax(h_d, 2)

        elif self.likelihood_type == 'bernoulli':
            mu_d = torch.sigmoid(h_d)

        # =====ELBO
        # RE
        if self.likelihood_type == 'categorical':
            RE = log_categorical(x, mu_d, num_classes=self.num_vals, reduction='sum', dim=-1).sum(-1)

        elif self.likelihood_type == 'bernoulli':
            RE = log_bernoulli(x, mu_d, reduction='sum', dim=-1)

        # KL
        KL_z_2 = 0.5 * (delta_mu_2 ** 2 + torch.exp(delta_log_var_2) - delta_log_var_2 - 1).sum(-1)
        KL_z_1 = 0.5 * (delta_mu_1 ** 2 / torch.exp(log_var_1) + torch.exp(delta_log_var_1) - \
                        delta_log_var_1 - 1).sum(-1)

        KL = KL_z_1 + KL_z_2

        error = 0
        if np.isnan(RE.cpu().detach().numpy()).any():
            print('RE {}'.format(RE))
            print('KL {}'.format(KL))
            error = 1
        if np.isnan(KL.cpu().detach().numpy()).any():
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

    def generate(self, batch_size=64,cond=None):
        # step 1
        z_2 = torch.randn(batch_size, self.L).to(self.device)
        # step 2
        h_1 = self.nn_z_1(z_2)
        mu_1, log_var_1 = torch.chunk(h_1, 2, dim=1)
        # step 3
        z_1 = self.reparameterization(mu_1, log_var_1)

        # step 4
        h_d = self.nn_x(z_1)

        if self.likelihood_type == 'categorical':
            b = batch_size
            d = h_d.shape[1] // self.num_vals
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

    def train(self, batch_data):
        loss = self.forward(*batch_data)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return loss.item()
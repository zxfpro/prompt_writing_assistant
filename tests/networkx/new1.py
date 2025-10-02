```python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# processor = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_emb, num_heads=8):
        super().__init__()
        
        # hyperparams
        self.D = num_emb
        self.H = num_heads
        
        # weights for self-attention
        self.w_k = nn.Linear(self.D, self.D * self.H)
        self.w_q = nn.Linear(self.D, self.D * self.H)
        self.w_v = nn.Linear(self.D, self.D * self.H)
        
        # weights for a combination of multiple heads
        self.w_c = nn.Linear(self.D * self.H, self.D)
            
    def forward(self, x, causal=True):
        # x: B(atch) x T(okens) x D(imensionality)
        B, T, D = x.size()
        
        # keys, queries, values
        k = self.w_k(x).view(B, T, self.H, D) # B x T x H x D
        q = self.w_q(x).view(B, T, self.H, D) # B x T x H x D
        v = self.w_v(x).view(B, T, self.H, D) # B x T x H x D
        
        k = k.transpose(1, 2).contiguous().view(B * self.H, T, D) # B*H x T x D
        q = q.transpose(1, 2).contiguous().view(B * self.H, T, D) # B*H x T x D
        v = v.transpose(1, 2).contiguous().view(B * self.H, T, D) # B*H x T x D
        
        k = k / (D**0.25) # scaling 
        q = q / (D**0.25) # scaling
        
        # kq
        kq = torch.bmm(q, k.transpose(1, 2)) # B*H x T x T
        
        # if causal
        if causal:
            mask = torch.triu_indices(T, T, offset=1)
            kq[..., mask[0], mask[1]] = float('-inf')
        
        # softmax
        skq = F.softmax(kq, dim=2)
        
        # self-attention
        sa = torch.bmm(skq, v) # B*H x T x D
        sa = sa.view(B, self.H, T, D) # B x H x T x D
        sa = sa.transpose(1, 2) # B x T x H x D
        sa = sa.contiguous().view(B, T, D * self.H) # B x T x D*H
        
        out = self.w_c(sa) # B x T x D
        
        return out  

class TransformerBlock(nn.Module):
    def __init__(self, num_emb, num_neurons, num_heads=4):
        super().__init__()
        
        # hyperparams
        self.D = num_emb
        self.H = num_heads
        self.neurons = num_neurons
        
        # components
        self.msha = MultiHeadSelfAttention(num_emb=self.D, num_heads=self.H)
        self.layer_norm1 = nn.LayerNorm(self.D)
        self.layer_norm2 = nn.LayerNorm(self.D)
        
        self.mlp = nn.Sequential(nn.Linear(self.D, self.neurons * self.D),
                                nn.GELU(),
                                nn.Linear(self.neurons * self.D, self.D))
    
    def forward(self, x, causal=True):
        # Multi-Head Self-Attention
        x_attn = self.msha(x, causal)
        # LayerNorm
        x = self.layer_norm1(x_attn + x)
        # MLP
        x_mlp = self.mlp(x)
        # LayerNorm
        x = self.layer_norm2(x_mlp + x)
        
        return x

class LossFun(nn.Module):
    def __init__(self,):
        super().__init__()

        self.loss = nn.NLLLoss(reduction='none')

    def forward(self, y_model, y_true, reduction='sum'):
        # y_model: B(atch) x T(okens) x V(alues)
        # y_true: B x T
        B, T, V = y_model.size()

        y_model = y_model.view(B * T, V)
        y_true = y_true.view(B * T,)

        loss_matrix = self.loss(y_model, y_true) # B*T

        if reduction == 'sum':
            return torch.sum(loss_matrix)
        elif reduction == 'mean':
            loss_matrix = loss_matrix.view(B, T)
            return torch.mean(torch.sum(loss_matrix, 1))
        else:
            raise ValueError('Reduction could be either `sum` or `mean`.')


class Transformer(nn.Module):
    def __init__(self, num_tokens, num_token_vals, num_emb, num_neurons, num_heads=2, dropout_prob=0.1, num_blocks=10, device='cpu'):
        super().__init__()
        
        # Remember, always credit the author, even if it's you ;)
        print('Transformer by JT.')
        
        # hyperparams
        self.device = device
        self.num_tokens = num_tokens
        self.num_token_vals = num_token_vals
        self.num_emb = num_emb
        self.num_blocks = num_blocks
        
        # embedding layer
        self.embedding = torch.nn.Embedding(num_token_vals, num_emb)
        
        # positional embedding
        self.positional_embedding = nn.Embedding(num_tokens, num_emb)
        
        # transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(num_emb=num_emb, num_neurons=num_neurons, num_heads=num_heads))
        
        # output layer (logits + softmax)
        self.logits = nn.Sequential(nn.Linear(num_emb, num_token_vals))
        
        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # loss function
        self.loss_fun = LossFun()
        
    def transformer_forward(self, x, causal=True, temperature=1.0):
        # x: B(atch) x T(okens)
        # embedding of tokens
        x = self.embedding(x) # B x T x D
        # embedding of positions
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0).to(self.device)
        pos_emb = self.positional_embedding(pos)
        # dropout of embedding of inputs
        x = self.dropout(x + pos_emb)
        
        # transformer blocks
        for i in range(self.num_blocks):
            x = self.transformer_blocks[i](x)
        
        # output logits
        out = self.logits(x)
        
        return F.log_softmax(out/temperature, 2)
    
    @torch.no_grad()
    def sample(self, batch_size=4, temperature=1.0):
        x_seq = np.asarray([[self.num_token_vals - 1] for i in range(batch_size)])

        # sample next tokens
        for i in range(self.num_tokens-1):
            xx = torch.tensor(x_seq, dtype=torch.long, device=self.device)
            # process x and calculate log_softmax
            x_log_probs = self.transformer_forward(xx, temperature=temperature)
            # sample i-th tokens
            x_i_sample = torch.multinomial(torch.exp(x_log_probs[:,i]), 1).to(self.device)
            # update the batch with new samples
            x_seq = np.concatenate((x_seq, x_i_sample.to('cpu').detach().numpy()), 1)
        
        return x_seq
    
    @torch.no_grad()
    def top1_rec(self, x, causal=True):
        x_prob = torch.exp(self.transformer_forward(x, causal=True))[:,:-1,:].contiguous()
        _, x_rec_max = torch.max(x_prob, dim=2)
        return torch.sum(torch.mean((x_rec_max.float() == x[:,1:].float().to(device)).float(), 1).float())
    
    def forward(self, x, causal=True, temperature=1.0, reduction='mean'):
        # get log-probabilities
        log_prob = self.transformer_forward(x, causal=causal, temperature=temperature)
        
        return self.loss_fun(log_prob[:,:-1].contiguous(), x[:,1:].contiguous(), reduction=reduction)





```

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt

class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)




class AttentionLayer(nn.Module):
    def __init__(self, n_hidden, n_heads, 
                 d_keys=None, d_values=None, mix=False ,q_n=1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (n_hidden//n_heads)
        d_values = d_values or (n_hidden//n_heads)

        self.inner_attention = FullAttention()
        self.query_projection = nn.Linear(n_hidden * q_n, d_keys * n_heads)
        self.key_projection = nn.Linear(n_hidden, d_keys * n_heads)
        self.value_projection = nn.Linear(n_hidden, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, n_hidden)
        self.n_heads = n_heads
        self.mix = mix
        self.activation = nn.ReLU()

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)
        out = self.out_projection(out)

        return out, attn


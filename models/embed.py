import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attn import FullAttention, AttentionLayer
import math 
from copy import deepcopy



class PositionalEmbedding(nn.Module):
    def __init__(self, n_hidden, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_hidden).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, n_hidden, 2).float() * -(math.log(10000.0) / n_hidden)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, n_in, n_hidden):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv =nn.Conv1d(in_channels=n_in, out_channels=n_hidden, kernel_size=3, padding=padding, padding_mode='replicate')#1)#
        self.linear = nn.Linear(n_in, n_hidden)

        # params = deepcopy(self.tokenConv.state_dict())
        # params['weight'] = params['weight'].reshape( n_hidden,n_in)
        # self.linear.load_state_dict(params)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')        

    def forward(self, x):
        # print('self.tokenConv.state_dict()')
        # print(self.tokenConv.state_dict())
        x=x.permute(0, 2, 1)
        x = self.tokenConv(x)
        x=x.transpose(1,2)
        
        # print('self.linear.state_dict()')
        # print(self.linear.state_dict())
        # x= self.linear(x)
        return x

class TTokenEmbedding(nn.Module):
    def __init__(self, n_in, n_hidden, n_l):
        super(TTokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv2d(in_channels=n_in, out_channels=n_hidden, 
                                    kernel_size=(3,n_l), padding=(padding,0), padding_mode='replicate')
        self.norm = nn.BatchNorm1d(n_hidden)
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
        self.linear=nn.Linear(n_l*n_hidden,n_hidden, bias=True)
        self.activation = nn.ReLU()       

    def forward(self, x):
        x=x.permute(0, 3, 1, 2)
        x = self.tokenConv(x)
        #x=x.permute(0, 2, 3, 1)
        #x=x.reshape([x.shape[0],x.shape[1],-1])
        x =x.squeeze(-1)
        x = self.activation(self.norm(x))
        x=x.permute(0, 2, 1)
        # x = self.maxPool(x)
        #x=self.activation(self.linear(x))
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, n_in, n_hidden):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(n_in, n_hidden).float()
        w.require_grad = False

        position = torch.arange(0, n_in).float().unsqueeze(1)
        div_term = (torch.arange(0, n_hidden, 2).float() * -(math.log(10000.0) / n_hidden)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(n_in, n_hidden)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, n_hidden, embed_type='fixed'):
        super(TemporalEmbedding, self).__init__()


        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        
        self.day_embed = Embed(day_size, n_hidden)
        self.weekday_embed = Embed(weekday_size, n_hidden)
        self.month_embed = Embed(month_size, n_hidden)
    
    def forward(self, x):
        x = x.long()
        
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return  weekday_x + day_x + month_x 



class DataEmbedding(nn.Module):
    def __init__(self, n_in, n_hidden, n_l=None,  value_type='1D',embed_type='fixed', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(n_in=n_in, n_hidden=n_hidden) if value_type=='1D' else TTokenEmbedding(n_in=n_in, n_hidden=n_hidden ,n_l=n_l)
        self.position_embedding = PositionalEmbedding(n_hidden=n_hidden)
        self.temporal_embedding = TemporalEmbedding(n_hidden=n_hidden, embed_type=embed_type) 

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, data_stamp):
        x = self.value_embedding(x) + self.position_embedding(x) #+ self.temporal_embedding(data_stamp)
        
        return self.dropout(x)


class PortraitEmbedding(nn.Module):
    def __init__(self, cate_fea_nuniqs, nume_fea_size=0, 
                 n_hidden=[256, 128], dropout=0.1): 
        """
        cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_fea_size: 数值特征的个数
        """
        super().__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size
        
        self.sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_nuniqs])  # 类别特征的表示
        self.linear_a=nn.Linear(self.cate_fea_size+self.nume_fea_size, n_hidden[0], bias=True)
        self.linear_b=nn.Linear(n_hidden[0], n_hidden[1], bias=True)
        self.norm = nn.BatchNorm1d(n_hidden[1])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X_sparse, X_dense):
        """
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入 [bs, dense_fea_size]
        """
        X_sparse = torch.tensor(X_sparse, dtype=torch.int64)
        X_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1) 
                             for i, emb in enumerate(self.sparse_emb)]
        X_sparse_res = torch.cat(X_sparse_res, dim=1)  # [bs, cate_fea_size]
        X_res=torch.cat([X_dense,X_sparse_res], dim=1)
        X_res = self.activation(self.linear_a(X_res)) 
        out = self.activation(self.norm(self.linear_b(X_res)))# 非元学习，batchnorm
        out = self.activation(self.linear_b(X_res))
        return self.dropout(out)





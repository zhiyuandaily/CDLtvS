import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from math import sqrt
from models.embed import TemporalEmbedding
from models.attn import AttentionLayer
from models.embed import DataEmbedding
import random

#TCN
# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size#这个chomp_size就是padding的值

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# 这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))#weight_norm()
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 =weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))# weight_norm()
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.norm = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2,self.norm)
        self.downsample = nn.Conv1d(n_inputs, n_outputs,kernel_size=1) if n_inputs != n_outputs else None
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in')      
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(res+out)

#最后就是TCN的主网络了
class TemporalConvNetOriginal(nn.Module):
    def __init__(self, num_inputs, num_channels, seq_len, kernel_size=2, dropout=0.2,if_pool_linear=True):
        super(TemporalConvNetOriginal, self).__init__()
        layers = []
        num_levels = len(num_channels)
        self.if_pool_linear=if_pool_linear
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        n_hidden= num_channels[-1]
        self.pool_linear=nn.Sequential(
            nn.Linear(n_hidden*seq_len,n_hidden, bias=True),
            nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, data_stamp=None):
        x = self.network(x.transpose(-1,1)).transpose(-1,1)
        # x= self.pool_linear(x[:,-1,:])#x.reshape(x.shape[0],-1))
        if self.if_pool_linear:
            x=self.pool_linear(x.reshape(x.shape[0],-1))
            return self.dropout(x)
        else:
            return x


# Informer
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        # x = x + self.dropout(self.attention(
        #     x, x, x,
        #     attn_mask = attn_mask
        # ))
        new_x, attn = self.attention(
            x, x, x,
            attn_mask = attn_mask
        )
        # x = self.dropout(new_x) + x
        # y = x = self.norm1(x)

        y = x  = self.norm1(new_x) +x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        
        

        return x+self.norm2(y), attn #self.norm2(x+y), attn  

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, mode='train'):
        # x [B, L, D]
        attns = []
        g=None
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                if isinstance(conv_layer,MEConvLayer):
                    x, g=conv_layer(x,mode)
                else:
                    x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns, g

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
    
class SEQInformer(nn.Module):
    def __init__(self, n_in, n_hidden=512, n_heads=8, e_layers=3, seq_len=30, 
                dropout=0.1,embed_type='fixed', distil=True, expert_n=1):
        super(SEQInformer, self).__init__()
        
        # Encoding
        self.enc_embedding = DataEmbedding(n_in, n_hidden, embed_type=embed_type, dropout=dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(n_hidden, n_heads, mix=False),
                    d_model=n_hidden,
                    d_ff=n_hidden,
                    dropout=dropout
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    c_in=n_hidden
                ) if l<(e_layers-2) or expert_n==1
                else MEConvLayer(
                    c_in=n_hidden,
                    seq_len=seq_len//(2**(e_layers-2)),
                    expert_n=expert_n
                )
                for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(n_hidden) if e_layers>1 else None
        )
        self.N_L=seq_len//(2**(e_layers-1))
        self.pool_linear=nn.Sequential(
            nn.Linear(n_hidden*self.N_L,n_hidden, bias=True),#
            nn.ReLU())
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_enc, x_mark_enc, mode='train'):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns, g = self.encoder(enc_out, mode)
        #enc_out=self.pool_linear(enc_out[:,-1,:])
        enc_out=self.pool_linear(enc_out.reshape(enc_out.shape[0],-1))
        scale=self.N_L
        enc_out=enc_out/scale

        return self.dropout(enc_out), g


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias=True, dropout=0.1, nonlinearity=None):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = getattr(nn, nonlinearity)() if nonlinearity else None
        self.reset_parameters()

    def forward(self, x):
        x = self.dropout(self.fc(x))
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)
        return x

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)

class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dims, out_dim=1, h_active="LeakyReLU", o_active="ReLU"):
        super(MLP, self).__init__()
        layers = [Dense(inp_dim, hidden_dims[0], nonlinearity=h_active)]
        for i in range(len(hidden_dims) - 1):
            layers.append(Dense(hidden_dims[i], hidden_dims[i + 1], nonlinearity=h_active))
        layers.append(Dense(hidden_dims[-1], out_dim, nonlinearity=o_active))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)

class DeepFM(nn.Module):
    def __init__(self, cate_fea_nuniqs, nume_fea_size=0, emb_size=8, 
                 hid_dims=[256, 128], num_classes=1, dropout=[0.2, 0.2]): 
        """
        cate_fea_nuniqs: 类别特征的唯一值个数列表，也就是每个类别特征的vocab_size所组成的列表
        nume_fea_size: 数值特征的个数，该模型会考虑到输入全为类别型，即没有数值特征的情况 
        """
        super().__init__()
        self.cate_fea_size = len(cate_fea_nuniqs)
        self.nume_fea_size = nume_fea_size-3
        
        """FM部分"""
        # 一阶
        if self.nume_fea_size != 0:
            self.fm_1st_order_dense = nn.Linear(self.nume_fea_size, 1)  # 数值特征的一阶表示
        self.fm_1st_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, 1) for voc_size in cate_fea_nuniqs])  # 类别特征的一阶表示
        
        # 二阶
        self.fm_2nd_order_sparse_emb = nn.ModuleList([
            nn.Embedding(voc_size, emb_size) for voc_size in cate_fea_nuniqs])  # 类别特征的二阶表示
        
        """DNN部分"""
        self.all_dims = [self.cate_fea_size * emb_size] + hid_dims
        self.dense_linear = nn.Linear(self.nume_fea_size, self.cate_fea_size * emb_size)  # 数值特征的维度变换到FM输出维度一致
        self.relu = nn.ReLU()
        # for DNN 
        for i in range(1, len(self.all_dims)):
            setattr(self, 'linear_'+str(i), nn.Linear(self.all_dims[i-1], self.all_dims[i]))
            setattr(self, 'batchNorm_' + str(i), nn.BatchNorm1d(self.all_dims[i]))
            setattr(self, 'activation_' + str(i), nn.ReLU())
            setattr(self, 'dropout_'+str(i), nn.Dropout(dropout[i-1]))
        # for output 
        self.dnn_linear = nn.Linear(hid_dims[-1], num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X_sparse, X_dense=None):
        X_dense=X_dense[:,:self.nume_fea_size]
        """
        X_sparse: 类别型特征输入  [bs, cate_fea_size]
        X_dense: 数值型特征输入（可能没有）  [bs, dense_fea_size]
        """
        
        """FM 一阶部分"""
        X_sparse = torch.tensor(X_sparse, dtype=torch.int64)
        fm_1st_sparse_res = [emb(X_sparse[:, i].unsqueeze(1)).view(-1, 1) 
                             for i, emb in enumerate(self.fm_1st_order_sparse_emb)]
        fm_1st_sparse_res = torch.cat(fm_1st_sparse_res, dim=1)  # [bs, cate_fea_size]
        fm_1st_sparse_res = torch.sum(fm_1st_sparse_res, 1,  keepdim=True)  # [bs, 1]
        
        if X_dense is not None:
            fm_1st_dense_res = self.fm_1st_order_dense(X_dense) 
            fm_1st_part = fm_1st_sparse_res + fm_1st_dense_res
        else:
            fm_1st_part = fm_1st_sparse_res   # [bs, 1]
        
        """FM 二阶部分"""
        fm_2nd_order_res = [emb(X_sparse[:, i].unsqueeze(1)) for i, emb in enumerate(self.fm_2nd_order_sparse_emb)]
        fm_2nd_concat_1d = torch.cat(fm_2nd_order_res, dim=1)  # [bs, n, emb_size]  n为类别型特征个数(cate_fea_size)
        
        # 先求和再平方
        sum_embed = torch.sum(fm_2nd_concat_1d, 1)  # [bs, emb_size]
        square_sum_embed = sum_embed * sum_embed    # [bs, emb_size]
        # 先平方再求和
        square_embed = fm_2nd_concat_1d * fm_2nd_concat_1d  # [bs, n, emb_size]
        sum_square_embed = torch.sum(square_embed, 1)  # [bs, emb_size]
        # 相减除以2 
        sub = square_sum_embed - sum_square_embed  
        sub = sub * 0.5   # [bs, emb_size]
        
        fm_2nd_part = torch.sum(sub, 1, keepdim=True)   # [bs, 1]
        
        """DNN部分"""
        dnn_out = torch.flatten(fm_2nd_concat_1d, 1)   # [bs, n * emb_size]
        
        if X_dense is not None:
            dense_out = self.relu(self.dense_linear(X_dense))   # [bs, n * emb_size]
            dnn_out = dnn_out + dense_out   # [bs, n * emb_size]
        
        for i in range(1, len(self.all_dims)):
            dnn_out = getattr(self, 'linear_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'batchNorm_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'activation_' + str(i))(dnn_out)
            dnn_out = getattr(self, 'dropout_' + str(i))(dnn_out)
        
        dnn_out = self.dnn_linear(dnn_out)   # [bs, 1]
        out = fm_1st_part + fm_2nd_part + dnn_out   # [bs, 1]
        out = self.relu(out)
        return out


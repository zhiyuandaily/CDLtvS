import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from models.encoder import DeepFM,SEQInformer,TemporalConvNetOriginal
import time
from math import sqrt


class CLVBase(nn.Module):
    def __init__(self,dense_fea_n, sparse_fea_nuniqs, 
                    seq_n, n_out, n_heads=8, n_hidden=256, n_embed_p=128,dropout=0.1,
                    seq_encoder_type='tcn',embed_type='fixed',expert_n=8):
        super(CLVBase, self).__init__()
        
        self.dense_fea_n=dense_fea_n
        self.expert_n=expert_n
        self.p_encoder=DeepFM(sparse_fea_nuniqs, dense_fea_n, emb_size=8, 
                 hid_dims=[n_hidden, n_hidden], num_classes=n_embed_p, dropout=[dropout, dropout])#PortraitEmbedding(cate_fea_nuniqs=sparse_fea_nuniqs, nume_fea_size=dense_fea_n, n_hidden=[n_hidden*2, n_embed_p])
        self.seq_encoder_type=seq_encoder_type
        
        n_hidden_mlp=n_hidden
        if self.seq_encoder_type=='lstm':
            n_hidden_mlp=n_hidden_mlp*30
            self.s_encoder = nn.LSTM(input_size=seq_n, hidden_size=n_hidden, num_layers=2)
        if self.seq_encoder_type=='tcn':
            self.s_encoder = TemporalConvNetOriginal(num_inputs=seq_n, num_channels=[n_hidden], seq_len=30,dropout=dropout) #if i<2 else LSTM2( seq_n1=seq_n1,n_hidden=n_hidden) for i in range(self.expert_n)])
        if self.seq_encoder_type=='informer':
            self.s_encoder = SEQInformer(n_in=seq_n, n_hidden=n_hidden, n_heads=n_heads, e_layers=2, seq_len=30,dropout=dropout, embed_type=embed_type, distil=True)
        self.fuse_linears=nn.Linear(n_hidden*2,n_hidden)
        self.out_linears =nn.Linear(n_hidden,n_out)
        
        # 多专家g
        self.gate_a=nn.Linear(n_hidden, n_hidden)#TemporalConvNet(num_inputs=seq_n1, num_channels=[n_hidden,n_hidden], seq_len=30,dropout=dropout)
        self.gate_b_smax=nn.Linear(n_hidden*1,self.expert_n) #n
        self.gate_b_smoid=nn.Linear(n_hidden*1,self.expert_n) #n
        nn.init.xavier_uniform_(self.gate_b_smax.weight)
        nn.init.xavier_uniform_(self.gate_b_smoid.weight)

        Print=False
        if Print:
            print(self.fuse_linears.weight)
            print(self.gate_b_smoid.weight)

        
        # print('self.gate_a.weight',self.gate_a.weight)
        # print('self.gate_b_smax.weight',self.gate_b_smax.weight)
    

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = nn.BatchNorm1d(n_hidden)

    
    def expert_g(self,x,mode,step_count=0):
        g_emb=F.tanh(self.gate_a(x.reshape([x.shape[0],-1])))
        g_emb_smax=g=self.gate_b_smax(g_emb)[:,:self.expert_n]
        g_emb_smoid=self.gate_b_smoid(g_emb)[:,:self.expert_n]
        index = torch.topk(g_emb_smax, k=int(1), dim=-1, largest=True)[1]
        # if mode=='train':
        #     #change_size=int(0.1*(1/(max(step_count-0,0)+1))*len(index))
        #     change_size=int(0.1*len(index))
        #     index_to_change = random.sample(range(len(index)), change_size)
        #     index[index_to_change]=torch.randint(0,self.expert_n,(change_size,1),device=g.device)
        mask = torch.zeros_like(g, device=g.device, requires_grad=False)
        mask.scatter_(-1, index, 1.)
        spase_g = torch.where(mask > 0, g, torch.full_like(g, float('-inf')))
        spase_g=F.softmax(spase_g)
        return spase_g, [F.softmax(g_emb_smax),F.sigmoid(g_emb_smoid)]

    
    def forward(self, data_p,data_stamp,data_s, mode='train',encode_type='',step_count=0):
        p_e=self.p_encoder(data_p[:,self.dense_fea_n:],data_p[:,:self.dense_fea_n])
        if encode_type!='':
            mode='test'
        if encode_type=='p':
            return p_e

        if self.seq_encoder_type=='lstm':
            S_e,_=self.s_encoder(data_s)
        if self.seq_encoder_type=='informer':
            S_e,_=  self.s_encoder(data_s,data_stamp)
        if self.seq_encoder_type=='tcn':
            S_e=  self.s_encoder(data_s,data_stamp)
        s_e =S_e.reshape([S_e.shape[0],-1])
        if encode_type=='s':
            return s_e
        
        e=self.activation(self.fuse_linears(torch.cat((s_e,p_e),dim=-1)))
        spase_g,g_list=self.expert_g(e,mode,step_count)
        spase_g=spase_g.unsqueeze(2).repeat(1,1,e.shape[1]//self.expert_n).reshape(e.shape[0],-1)
        e= e*spase_g
        if encode_type=='sp':
            p_e=p_e.detach()
            s_e=s_e.detach()
            # g_list[0]=g_list[0].detach()
            return p_e,s_e,e,g_list
        else:
            out=self.out_linears(e) # 激活函数放loss
            return out,g_list




# class MixCrossModule(nn.Module):
#     def __init__(self,seq_n, n_heads=8, n_hidden=256, n_embed_p=128,dropout=0.1,
#                     seq_encoder_type='informer',embed_type='fixed',cluster_n=1):
#         super(MixCrossModule, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.seq_encoder_type=seq_encoder_type
#         self.cluster_n=cluster_n
#         n_hidden_mlp=n_hidden

#         self.query_layer=nn.Sequential(
#             nn.Linear(n_hidden*3,n_hidden),
#             nn.ReLU(),
#             nn.BatchNorm1d(n_hidden)
#         )


#         self.key = torch.nn.Parameter(torch.randn((n_hidden*1,cluster_n),requires_grad=True))
#         torch.nn.init.normal_(self.key, mean=0.0, std=1.0)
#         self.register_parameter("key",self.key)
#         self.value = torch.nn.Parameter(torch.randn((cluster_n,n_hidden,n_hidden),requires_grad=True))
#         torch.nn.init.normal_(self.value, mean=0.0, std=0.1)
#         self.register_parameter("value",self.value)

        
#         self.mapping_weight = torch.nn.Parameter(torch.randn((n_hidden,n_hidden),requires_grad=True))
#         torch.nn.init.normal_(self.mapping_weight, mean=0.0, std=1.0)
#         self.register_parameter("mapping_weight",self.mapping_weight)

#         self.activation = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         self.mapping_layer=nn.Sequential(
#             nn.Linear(n_hidden,n_hidden*n_hidden),
#             nn.ReLU()
#         )

#     def forward(self, p_1, p_2, s_1,s_2, e_2):
#         query=self.query_layer(torch.cat((p_1,p_2,s_2),dim=1))
#         # 基于内存的
#         scores = torch.cosine_similarity(query.unsqueeze(2).repeat(1,1,self.cluster_n),self.key[:,:self.cluster_n].unsqueeze(0).repeat(query.shape[0],1,1),dim=1)
#         scale = 1./sqrt(self.key.shape[1])
#         if self.key.shape[1]!=1:
#             attn = self.dropout(torch.softmax(scores* scale, dim=-1))
#         else:
#             attn = torch.softmax(scores* scale, dim=-1)

#         mapping = torch.einsum("bd,dhk->bhk", attn, self.value[:self.cluster_n,:,:])
#         # # 基于个性化参数的
#         mapping=mapping+self.mapping_weight.unsqueeze(0).repeat(mapping.shape[0],1,1)
#         out =  torch.bmm(e_2.unsqueeze(1), mapping).squeeze(1)#s_2*mapping#
#         out=self.activation(out)
#         return self.dropout(out)


class MixExpert(nn.Module):
    def __init__(self, root_path, dense_fea_n1, sparse_fea_nuniqs1, dense_fea_n2, sparse_fea_nuniqs2, 
                    seq_n1, seq_n2, item_seq_l,n_out,
                    dropout=0.1, n_heads=8, n_hidden=256, n_embed_p=16,
                    seq_encoder_type='informer',embed_type='fixed',expert_n=4,cluster_n=8):
        super(MixExpert, self).__init__()
        self.expert_n=expert_n
        n_out=2
        torch.manual_seed(4765)
        self.tgt_model = CLVBase(dense_fea_n1, sparse_fea_nuniqs1,seq_n1, n_out, n_heads, n_hidden, n_embed_p,dropout, seq_encoder_type,embed_type,expert_n)
        torch.manual_seed(4765)
        self.src_model = CLVBase(dense_fea_n2, sparse_fea_nuniqs2, seq_n2, n_out, n_heads, n_hidden, n_embed_p,dropout, seq_encoder_type,embed_type,expert_n)
        torch.manual_seed(4765)
        # self.cross_model=MixCrossModule(3, n_heads, n_hidden, n_embed_p,dropout,
        #     seq_encoder_type,embed_type,cluster_n)
        self.cross_model=nn.Sequential( 
            nn.Linear(n_hidden,n_hidden, bias=True),
            nn.ReLU())

        self.out_layer=nn.Sequential(
            nn.Linear(n_hidden*2,80, bias=True),
            nn.ReLU(),
            nn.Linear(80,40, bias=True),
            nn.ReLU(),
            nn.Linear(40,1, bias=True))
        self.out_g=nn.Sequential( 
            nn.Linear(n_hidden*2,80, bias=True),
            nn.ReLU(),
            nn.Linear(80,1, bias=True),nn.Sigmoid())

        self.align_W_cross = nn.Sequential( 
            nn.Linear(n_hidden,80, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(80))
        self.align_W_1 = nn.Sequential( 
            nn.Linear(n_hidden,80, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(80))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    

    def forward(self, data_p1, data_p2, data_stamp, data_s1,data_s2_wn , data_s2_wi, stage='', mode='train',step_count=0):
        if stage in ['src']:
            out,g_list=self.src_model(data_p2,data_stamp,data_s2_wn, mode,encode_type='',step_count=step_count)
            return out,g_list
        elif stage in ['tgt']:
            out,g_list=self.tgt_model(data_p1,data_stamp,data_s1, mode,encode_type='',step_count=step_count)
            return out,g_list
        elif stage in ['meta']:
            p_2,s_2,e_2,g_list2=self.src_model.forward(data_p2,data_stamp,data_s2_wn, mode=mode,encode_type='sp',step_count=step_count)
            p_1,s_1,e_1,g_list1=self.tgt_model.forward(data_p1,data_stamp,data_s1, mode=mode,encode_type='sp',step_count=step_count)
            cross_e=self.cross_model(e_2)
            out=self.out_layer(torch.cat((e_1,cross_e),dim=-1))

            #跨域信息对齐
            last_clv=torch.sum(data_s1[:,:,1],dim=1)
            align_info=[last_clv,self.align_W_cross(cross_e),self.align_W_1(e_1)]
            return out,g_list1+align_info
        



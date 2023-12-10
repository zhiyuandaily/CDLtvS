import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class MyLoss(nn.Module):
    def __init__(self,model,args):
        super().__init__()
        self.model=model
        self.args=args

        if args.loss_type=='SmoothL1Loss':
            self.LVPLoss = nn.SmoothL1Loss(beta=10)
        elif args.loss_type=='L1Loss':
            self.LVPLoss = nn.L1Loss()
        elif args.loss_type=='Zhou':
            self.LVPLoss = class_trans_loss

    def forward(self, preds1, targets1,  preds2=None, targets2=None,trend_d2=None,stage='tgt'):
        if self.args.loss_type=='Zhou':
            ltv_loss, loss_list = self.LVPLoss([preds1]+preds2, targets1[:,0],self.args.lamdas,self.args.step_count,stage)
            return ltv_loss, torch.tensor(loss_list)
        else:
            loss=self.LVPLoss(preds1, targets1[:,0])
            return loss, torch.tensor([loss]) 


# ZhouLoss
def class_trans_pred(preds1,preds2):
    bias_prob=preds1
    bias_prob=torch.tensor(bias_prob)
    # log l1Loss
    pred=F.relu(bias_prob[:,0:1])
    return np.array(pred)

def class_trans_loss(logits,labels,lamdas,step_count,stage='tgt'):
    if stage=='meta':
        bias_prob,g_d_smax,g_d_smoid,last_clv,cross_s_align,s_1=logits#=logits[0]#,p_2,p_1
    else:
        bias_prob,g_d_smax,g_d_smoid=logits
    N=g_d_smoid.shape[0]
    expert_n=g_d_smoid.shape[1]

    # 0.预测损失
    # 判断是否为0的loss,无效
    positive = torch.where(labels>0,1,0).float()
    # L1
    loss_bias=F.l1_loss(F.relu(bias_prob[:,0:1]),labels)#labels)#torch.log(labels+1))#

    # 1.负载均衡
    if stage!='meta':#'=='meta2': #
        probi=torch.mean(g_d_smax,dim=0)
        max_i=torch.argmax(g_d_smax,dim=1)
        indices,counts =max_i.unique(return_counts=True)
        zeros = torch.zeros_like(probi, requires_grad=True)
        N=torch.sum(counts)
        freqs=counts/N
        freqs = zeros.scatter(0, indices, freqs).detach()
        mean_bias=torch.ones_like(probi, requires_grad=False)/probi.shape[0]
        mean_bias=torch.sum(mean_bias*mean_bias,dim=0)
        load_balancing_loss = (torch.sum(probi*freqs,dim=0))*freqs.shape[0]#(torch.sum(probi*freqs,dim=0)-mean_bias)*N
        load_balancing_loss=0

        # 2.0分类损失
        _, idx = labels.squeeze().sort(0, descending=False)#大小为[batch size, num_classes*top_k]
        _, rank = idx.sort(0)#再对索引升序排列，得到其索引作为排名rank
        positive= torch.zeros_like(g_d_smoid, requires_grad=False)
        for i in range(len(rank)):
            j=int(rank[i]//max(int(N//expert_n),1))
            positive[i,:j+1]=1
            
        g_d_smoid_label=positive
        loss_smoid = F.binary_cross_entropy(g_d_smoid,g_d_smoid_label)
        # 2.1分类损失
        max_i =(torch.sum(g_d_smoid_label,dim=1)-1).long().detach()
        zeros = torch.zeros_like(g_d_smax, requires_grad=True)
        g_d_smax_label = zeros.scatter(1, max_i.unsqueeze(1), 1).long()
        loss_smax = F.nll_loss(torch.log(g_d_smax),max_i)


    else:
        load_balancing_loss=0
        loss_smoid=0
        loss_smax=0

    
    
    # 2.跨域对齐损失
    if stage=='meta':
        align_positive=torch.where(last_clv>=0,1,0).detach()
       
        
        margin=torch.FloatTensor([-1]).to(align_positive.device)
        pdist = nn.PairwiseDistance(p=2)
        p_sim = torch.cosine_similarity(cross_s_align,s_1,dim=-1)
       
        align_loss_t=((1-p_sim)*align_positive).sum()/align_positive.sum()



        align_negtive=1-align_positive
        align_loss_n=p_sim-margin
        align_loss_n=torch.where(align_loss_n>0,align_loss_n,torch.zeros_like(align_loss_n).float())
        if align_negtive.sum()!=0:
            align_loss_n=(align_loss_n*align_negtive).sum()/align_negtive.sum()
        else:
            align_loss_n=0
        align_loss=align_loss_t+align_loss_n

        in_loss=InfoNCE(negative_mode='unpaired')#InfoNCELoss()
        positive_index=torch.tensor([i for i in range(align_positive.size(0)) if align_positive[i]==1]).to(cross_s_align.device)
        positive_cross_s_align=torch.index_select(cross_s_align,0,positive_index)
        positive_s_1=torch.index_select(s_1,0,positive_index)
        align_loss_intra_1=in_loss(positive_cross_s_align,positive_s_1)

    else:
        align_loss=0
        align_loss_intra_1=0
        align_loss_intra_2=0
    
   
    lamdas=lamdas.copy()
    for i in range(len(lamdas)):
        lamdas[i]=max(lamdas[i]*0.01,lamdas[i]/(step_count*5+1))
    return loss_bias+\
        loss_smax*lamdas[0]+loss_smoid*lamdas[1]+\
            align_loss*lamdas[2]+align_loss_intra_1*lamdas[3],[loss_bias,loss_smax,loss_smoid,align_loss,align_loss_intra_1]#loss_smax+loss_smoid*10 #+classification_loss



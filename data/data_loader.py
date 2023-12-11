import os
import numpy as np
import pandas as pd
import pickle
import pickle
import gc

import torch
import copy
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from sklearn.mixture import GaussianMixture

#from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Custormer_Data():
    def __init__(self, args):
        self.args=args
        self.data_partition=[0.4,0.6]
        self.set_partition=0.5
        self.time_window=30
        self.scaler_list = [StandardScaler() for i in range(4)]
        self.item_third_cate_id_len=7515
        self.unqualified_custormer_n=0 #不合格商家个数
        self.corr_array=[]
        self.add_RFM=False

  

    def delete_unqualified_custormer(self,data1, data2):
        for cusid in data1['cusid'].unique():
            temp_data1=data1[data1['cusid']==cusid]
            temp_data2=data2[data2['cusid']==cusid]
            length1=len(temp_data1)
            length2=len(temp_data2)
            # 去除源域时序数据无规律
            self_corr=np.corrcoef(temp_data2.loc[7:,'num'].values,temp_data2.loc[:length2-8,'num'])[0,1]
            self_corr=abs(self_corr)
            if not np.isnan(self_corr):
                self.corr_array.append(self_corr)
            if min(length1, length2) < self.time_window*3:
                self.unqualified_custormer_n=self.unqualified_custormer_n+1
                data1=data1[data1['cusid']!=cusid]
            elif np.isnan(self_corr):
                data1=data1[data1['cusid']!=cusid]
            elif self_corr<0.1:
                data1=data1[data1['cusid']!=cusid]
            else:
                continue
        return data1

    def __read_data__(self):
        # 0. 筛选城市
        trader_code_city_pd=pd.read_pickle('/home/zzy17/商家价值/RAWDATA/pop_pd_by_cdm_cust_ldop_trader_da.pkl')
        if self.args.data_info=='GuangZhou':
            trader_code_city_pd=trader_code_city_pd[trader_code_city_pd['sign_city_id']==1601]
        elif self.args.data_info=='BeiJing':
            trader_code_city_pd=trader_code_city_pd[trader_code_city_pd['sign_province_id']==1]
        trader_code_city_pd=trader_code_city_pd['trader_code']
        # 1. 客户时序数据读取
        data1=pd.DataFrame()
        data2=pd.DataFrame()
        data2_plus=pd.DataFrame()
        for i in range(self.args.data_volume):
            temp_df = pd.read_pickle('/home/zzy17/商家价值/DATASET/trader_waybill_pd/trader_waybill_pd_'+str(i)+'.pkl')[['trader_code','waybill_num','waybill_weight','start_date']]
            data1 =data1.append(temp_df[temp_df['trader_code'].isin(trader_code_city_pd)])
            temp_df = pd.read_pickle('/home/zzy17/商家价值/DATASET/trader_order_num_pd/trader_order_num_pd_'+str(i)+'.pkl')[['trader_code','sale_ord_num','sale_qtty','start_date']]
            data2 =data2.append(temp_df[temp_df['trader_code'].isin(trader_code_city_pd)])
            temp_df = pd.read_pickle('/home/zzy17/商家价值/DATASET/trader_order_item_pd/trader_order_item_pd_'+str(i)+'.pkl')[['trader_code','item_third_cate_cd','sale_ord_num','sale_qtty','start_date']]
            data2_plus =data2_plus.append(temp_df[temp_df['trader_code'].isin(trader_code_city_pd)])
        data1.columns = ['cusid','num','num2','date']
        data2.columns = ['cusid','num','num2','date']
        data2_plus.columns = ['cusid','item_third_cate_cd','num','num2','date']
        print('商家数量：',data2_plus['cusid'].nunique(),'品类数量：',data2_plus['item_third_cate_cd'].nunique())
        print('品类分布：',data2_plus['item_third_cate_cd'].value_counts()/len(data2_plus))
        

        print(data1.nunique())
        print(data2.nunique())
        # data = pd.concat([data1,data2])
        if not self.args.debug:
            data1 = self.delete_unqualified_custormer(data1,data2) # 去除不合格商家
        customer_id = data1[['cusid']].drop_duplicates().reindex()
        customer_id['customer_id'] = np.arange(len(customer_id))

        if self.args.debug:
            customer_id=customer_id.iloc[:50,:]

        data1 = pd.merge(data1, customer_id, on=['cusid'], how='inner')
        data2 = pd.merge(data2, customer_id, on=['cusid'], how='inner')
        data2_plus = pd.merge(data2_plus, customer_id, on=['cusid'], how='inner')
        data1 = data1[['customer_id','date','num','num2']]
        data2 = data2[['customer_id','date','num','num2']]
        item_third_cate_id=np.load('/home/zzy17/商家价值/MODEL/CLVP2023-main/ITEM_ENCODER/results/item_third_cate_id.npy',allow_pickle=True)
        item_third_cate_id=pd.DataFrame(item_third_cate_id)
        item_third_cate_id.columns=['item_third_cate_cd','item_third_cate_id']
        data2_plus=pd.merge(data2_plus, item_third_cate_id, on=['item_third_cate_cd'], how='inner')
        item_emb_arr=self.item_encoder.encode(torch.from_numpy(item_third_cate_id['item_third_cate_id'].values.astype(int))).detach().numpy()
        self.args.item_seq_l=data2_plus.groupby(['customer_id','date']).nunique().reset_index().item_third_cate_id.quantile(q=0.75, interpolation='nearest')
        self.args.item_seq_l=(self.args.item_seq_l+1)*30+1

        # 2. 客户画像特征读取
        portrait1=pd.read_pickle('/home/zzy17/商家价值/DATASET/waybill_customer_portrait.pkl')
        portrait2=pd.read_pickle('/home/zzy17/商家价值/DATASET/order_customer_portrait.pkl')
        portrait1_dense_features = [f for f in portrait1.columns.tolist() if f[0] == "I"]
        portrait1_sparse_features = [f for f in portrait1.columns.tolist() if f[0] == "C"]
        portrait2_dense_features = [f for f in portrait2.columns.tolist() if f[0] == "I"]
        portrait2_sparse_features = [f for f in portrait2.columns.tolist() if f[0] == "C"]
        self.args.dense_fea_n1=len(portrait1_dense_features)
        self.args.sparse_fea_nuniqs1 = [portrait1[f].nunique() for f in portrait1_sparse_features]
        self.args.dense_fea_n2=len(portrait2_dense_features)
        self.args.sparse_fea_nuniqs2 = [portrait2[f].nunique() for f in portrait2_sparse_features]
        if self.add_RFM==True:
            self.args.dense_fea_n1=self.args.dense_fea_n1+3
            self.args.dense_fea_n2=self.args.dense_fea_n2+3

            

        portrait1 = pd.merge(portrait1, customer_id, left_on='trader_code', right_on='cusid', how='inner')
        portrait2 = pd.merge(portrait2, customer_id, left_on='trader_code', right_on='cusid', how='inner')
        portrait1=pd.concat([portrait1[portrait1_dense_features],portrait1[portrait1_sparse_features],portrait1[['customer_id']]],axis=1)
        portrait2=pd.concat([portrait2[portrait2_dense_features],portrait2[portrait2_sparse_features],portrait1[['customer_id']]],axis=1)
        
        return customer_id, data1,data2,data2_plus,item_emb_arr,portrait1,portrait2
    
    def pad_item_sequence(self,item_seq, max_len=None):
        #0 [PAD], 1 [CLS], 2 [SEP]
        PAD_IDX = 0
        CLS_IDX = 1
        SEP_IDX = 2
        out_tensors=torch.tensor([CLS_IDX])
        position_ids=torch.tensor([0])
        if max_len is None:
            max_len = self.args.item_seq_l
        else:
            self.args.item_seq_l =max_len
        i=0
        for item_list in item_seq:
            # -item信息
            item_tensor=torch.tensor(item_list)+3
            out_tensors = torch.cat([out_tensors,item_tensor, torch.tensor([SEP_IDX])], dim=0)
            position_ids = torch.cat([position_ids,torch.ones(item_tensor.shape[0]+1)*i],dim=0)
            i=i+1
        if out_tensors.shape[0] < max_len:
                padding_len=max_len - out_tensors.shape[0]
                out_tensors = torch.cat([ out_tensors, torch.ones(padding_len)*PAD_IDX], dim=0)
                position_ids=torch.cat([position_ids,torch.zeros(padding_len)],dim=0)
        else:
            out_tensors = out_tensors[:max_len]
            out_tensors[max_len-1]=SEP_IDX
            position_ids= position_ids[:max_len]
        out_tensors=torch.cat([out_tensors.unsqueeze(1),position_ids.unsqueeze(1)],dim=1)
        return out_tensors.float() 

    def get_RFM(self,num_list):
        num_list = num_list[:self.time_window]
        days = [0 + i for i in range(len(num_list)) if num_list[i] > 0] + [0]
        r = self.time_window - max(days) - 1
        f = len(days) - 1
        m = sum(num_list)
        return r, f, m

    def get_data(self):
        CACHE_SIZE=20000
        SEED =100
        rng = np.random.RandomState(SEED)
        # 读取缓存数据
        if self.args.use_cache_data:
            path=self.args.root_path+'/data/'
            path=path+'non_meta'
            if self.args.data_normalization:
                path=path+'_normal'
            else:
                path=path+'_non_normal'
            if self.args.debug:
                path=path+'_debug_data'
            else:
                path=path+'_non_debug_data_'+str(self.args.data_volume)+str(self.args.data_info)
            with open(path+'/config.pkl', 'rb') as f: 
                [self.args.dense_fea_n1, self.args.sparse_fea_nuniqs1, self.args.dense_fea_n2, self.args.sparse_fea_nuniqs2, self.args.item_seq_l]=pickle.load(f)
                f.close()
            train_set= []
            test_set = []
            vali_set = []
            files = os.listdir(path)
            print('读取数据路径：',path)
            for file in files:
                if 'train_set_' in file:
                    with open(path+'/'+file, 'rb') as f: 
                        train_set.extend(pickle.load(f))
                        f.close()
                if 'vali_set_' in file:
                    with open(path+'/'+file, 'rb') as f: 
                        vali_set.extend(pickle.load(f))
                        f.close()
                elif 'test_set_' in file:
                    with open(path+'/'+file, 'rb') as f: 
                        test_set.extend(pickle.load(f))
                        f.close()
            # with open(path, 'rb') as f:  
            #     [self.args.dense_fea_n1, self.args.sparse_fea_nuniqs1, self.args.dense_fea_n2, self.args.sparse_fea_nuniqs2,self.args.item_seq_l, train_set, test_set]=pickle.load(f)
            #     f.close()
            # f = open(path,"rb+")
            # [self.args.dense_fea_n1, self.args.sparse_fea_nuniqs1, self.args.dense_fea_n2, self.args.sparse_fea_nuniqs2,self.args.item_seq_l, train_set, test_set]=pickle.load(f)   
            # f.close()
            return train_set, vali_set, test_set

        # 重新生成数据
        customer_id,data1,data2,data2_plus,item_emb_arr,portrait1,portrait2=self.__read_data__()

        
        train_set, vali_set, test_set= [], [], []

        
        for i in range(len(customer_id)):
            temp_data1 = data1[data1['customer_id']==i]
            temp_data1 = temp_data1.sort_values(['date'])
            length1 = len(temp_data1)
            temp_data1.index = range(length1)

            temp_data2 = data2[data2['customer_id']==i]
            temp_data2 = temp_data2.sort_values(['date'])
            # length2 = len(temp_data2)
            # temp_data2.index = range(length2)

            temp_data2_plus = data2_plus[data2_plus['customer_id']==i]

            # 增加距离第一次发货的时间信息
            temp_data1['fd_num']=temp_data1.index
            temp_data2['fd_num']=temp_data2.index

            # 经营品类信息--定义
            temp_data2_item_num=temp_data2_plus[['date','item_third_cate_id']].groupby('date').nunique().reset_index() #经营品类总数
            temp_data2_item_seq=temp_data2_plus[['date','item_third_cate_id','num']].sort_values(by=['num'],ascending=False).groupby('date').agg({'item_third_cate_id':list,'num':list}).reset_index() #经营品类列表（降序排序）
            
            # 经营品类信息--时间对齐
            temp_data2_item_num=pd.merge(temp_data2_item_num,temp_data2[['date']],on='date',how='right')
            temp_data2_item_num=temp_data2_item_num.sort_values(['date'])
            temp_data2_item_num=temp_data2_item_num.fillna(method='ffill').fillna(method='bfill')
            temp_data2_item_num=temp_data2_item_num.drop(labels=['date'], axis=1)
            temp_data2_item_seq=pd.merge(temp_data2_item_seq,temp_data2[['date']],on='date',how='right')
            temp_data2_item_seq=temp_data2_item_seq.sort_values(['date'])
            temp_data2_item_seq['item_third_cate_id']=temp_data2_item_seq['item_third_cate_id'].fillna(method='ffill').fillna(method='bfill')
            #temp_data2_item_seq['num']=temp_data2_item_seq['num'].fillna(0)
            temp_data2_item_seq=temp_data2_item_seq.drop(labels=['date','num'], axis=1)
            
            # 经营品类信息--合并temp_data2_item_num到temp_data2; 根据temp_data2_item_seq，item_emb_arr生成 temp_data2_plus
            temp_data2=pd.concat([temp_data2.reset_index(),temp_data2_item_num.reset_index()],axis=1)
            temp_data2=temp_data2.drop(labels=['index'], axis=1) 
            temp_data2=temp_data2.reset_index().drop(labels=['index'], axis=1) 
            length2 = len(temp_data2)
            temp_data2.index = range(length2)
            #temp_data2_plus=self.pad_item_sequence(temp_data2_item_seq.values.tolist(),item_emb_arr)
            temp_data2_plus = temp_data2_item_seq['item_third_cate_id'].values.tolist()

            node_id=i
            temp_portrait1=portrait1[portrait1['customer_id']==i]
            temp_portrait1 = temp_portrait1.drop(labels=['customer_id'], axis=1) 
            temp_portrait1 = temp_portrait1.iloc[0,:]
            temp_portrait1=np.append(temp_portrait1,node_id)
            temp_portrait2=portrait2[portrait2['customer_id']==i]
            temp_portrait2 = temp_portrait2.drop(labels=['customer_id'], axis=1) 
            temp_portrait2 = temp_portrait2.iloc[0,:]
            temp_portrait2=np.append(temp_portrait2,node_id)
            


            data_stamp = time_features(temp_data2[['date']], timeenc=0, freq='d')

            # # 补齐物流空白数据
            # padding_n=length2-length1
            # padding_temp_customer=temp_data2[['customer_id','date','num','num2','fd_num']].iloc[:padding_n].copy()
            # padding_temp_customer.iloc[:,padding_temp_customer.columns.get_loc("date")+1:]=0
            # padding_temp_customer['num_flag']=0 #当前没有过物流信息标识为0
            # temp_data1['num_flag']=1 #有物流信息标识为1
            # temp_data1=padding_temp_customer.append(temp_data1).reset_index()
            # temp_data1=temp_data1[['customer_id','date','num','num_flag']]#[['customer_id','date','num','num2','num_flag','fd_num']]
            # months=length2//self.time_window

            # 舍弃物流空白数据
            delete_n=length2-length1
            temp_data1['num_flag']=1 #有物流信息标识为1
            temp_data1=temp_data1[['customer_id','date','num','num_flag']]#[['customer_id','date','num','num2','num_flag','fd_num']]
            months=length1//self.time_window
            temp_data2=temp_data2[['customer_id','date','num']].iloc[delete_n:,:].reset_index()
            temp_data2_plus=temp_data2_plus[delete_n:]

            if self.add_RFM==True:
                r1,f1,m1=self.get_RFM(temp_data1['num'])
                r2,f2,m2=self.get_RFM(temp_data2['num'])
                temp_portrait1=np.concatenate((temp_portrait1[:self.args.dense_fea_n1-3],[r1,f1,m1],temp_portrait1[self.args.dense_fea_n1-3:]),axis=0)
                temp_portrait2=np.concatenate((temp_portrait2[:self.args.dense_fea_n2-3],[r2,f2,m2],temp_portrait2[self.args.dense_fea_n2-3:]),axis=0)
            
            support_set_indices=rng.choice(list(range(months)), size = min(int(months*self.set_partition), months-1), replace=False)
            print("随机选择支持集",support_set_indices)

            for j in range(months):
                end_j=min(length2,(j+2)*self.time_window)
                temp_data2_plus_j=self.pad_item_sequence(temp_data2_plus[j*self.time_window:(j+1)*self.time_window])
                label1=temp_data1.loc[(j+1)*self.time_window:end_j,'num'].sum()
                label2=temp_data2.loc[(j+1)*self.time_window:end_j,'num'].sum()
                old_sum_num2=temp_data2.loc[j*self.time_window:(j+1)*self.time_window,'num'].sum()
                trend_2= 0 if label2<old_sum_num2 else 1
                label1=[label1,trend_2]
                label2=[label2,trend_2]
                data=(list(temp_portrait1), list(temp_portrait2),
                                    list(data_stamp[j*self.time_window:(j+1)*self.time_window]),
                                    temp_data1.iloc[j*self.time_window:(j+1)*self.time_window,temp_data1.columns.get_loc("date")+1:].values.tolist(), 
                                    temp_data2.iloc[j*self.time_window:(j+1)*self.time_window,temp_data2.columns.get_loc("date")+1:].values.tolist(), 
                                    temp_data2_plus_j,
                                    label1, label2)
                if i<len(customer_id)*self.data_partition[0]: #训练集客户
                    train_set.append(data)
                elif j in support_set_indices:#j< min(months*self.set_partition, months-1): # 测试集、验证集客户，训练数据
                    train_set.append(data)
                elif i<len(customer_id)*self.data_partition[1]:
                    vali_set.append(data)
                else: #测试集客户，测试数据
                    test_set.append(data)
        print('客户数量:',len(customer_id))
        customer_id,data1,data2,data2_plus,item_emb_arr,portrait2=[],[],[],[],[],[]
        del customer_id,data1,data2,data2_plus,item_emb_arr,portrait2
        gc.collect()

        if self.args.data_normalization:
            train_set, vali_set, test_set=self.data_normalization(train_set, vali_set, test_set)
        else:
            self.node_features1=portrait1
            self.node_features2=portrait2
        path=self.args.root_path+'/data/'
        path=path+'non_meta'
        if self.args.data_normalization:
            path=path+'_normal'
        else:
            path=path+'_non_normal'
        if self.args.debug:
            path=path+'_debug_data'
        else:
            path=path+'_non_debug_data_'+str(self.args.data_volume)+str(self.args.data_info)

        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(path+'/graph1')
            os.makedirs(path+'/graph2')
        else:
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))
        with open(path+'/config.pkl', 'wb') as f: 
            pickle.dump([self.args.dense_fea_n1, self.args.sparse_fea_nuniqs1, self.args.dense_fea_n2, self.args.sparse_fea_nuniqs2, self.args.item_seq_l],f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
        
        #图节点特征处理
        self.node_features1=pd.DataFrame(self.node_features1)
        cname=self.node_features1.columns[-1]
        self.node_features1.rename(columns={cname:'node_id'},inplace=True)
        self.node_features1=self.node_features1.drop_duplicates(['node_id'])
        self.node_features1=self.node_features1.sort_values(['node_id'])
        print('节点个数',len(self.node_features1))
        print('节点ID范围', self.node_features1['node_id'].min(), self.node_features1['node_id'].max())
        build_graph(self.node_features1,self.args.dense_fea_n1, self.args.sparse_fea_nuniqs1, graph_path=path+'/graph1')

        self.node_features2=pd.DataFrame(self.node_features2)
        cname=self.node_features2.columns[-1]
        self.node_features2.rename(columns={cname:'node_id'},inplace=True)
        self.node_features2=self.node_features2.drop_duplicates(['node_id'])
        self.node_features2=self.node_features2.sort_values(['node_id'])
        print('节点个数',len(self.node_features2))
        print('节点ID范围', self.node_features2['node_id'].min(), self.node_features2['node_id'].max())
        build_graph(self.node_features2,self.args.dense_fea_n2, self.args.sparse_fea_nuniqs2, graph_path=path+'/graph2')

        for i in range(0,len(train_set),CACHE_SIZE):
            with open(path+'/train_set_'+str(i)+'.pkl', 'wb') as f: 
                pickle.dump(train_set[i: i+CACHE_SIZE ],f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
        for i in range(0,len(vali_set),CACHE_SIZE):
            with open(path+'/vali_set_'+str(i)+'.pkl', 'wb') as f: 
                pickle.dump(vali_set[i : i+CACHE_SIZE],f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
        for i in range(0,len(test_set),CACHE_SIZE):
            with open(path+'/test_set_'+str(i)+'.pkl', 'wb') as f: 
                pickle.dump(test_set[i : i+CACHE_SIZE],f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()
    
        # f = open(path,"wb")            
        # pickle.dump([self.args.dense_fea_n1, self.args.sparse_fea_nuniqs1, self.args.dense_fea_n2, self.args.sparse_fea_nuniqs2, self.args.item_seq_l, train_set, test_set],f)   
        # f.close()
        
        
        return train_set, vali_set, test_set

    def data_normalization(self, train_set, vali_set, test_set):
        # 定义需要归一化的变量的位置，最后储存归一化信息到四个变量data_target[0,1,2,3]
        portrait1_index=0
        portrait2_index=1
        seq1_index=3
        seq2_index=4
        
        data_target=[[] for i in range(4)]
        for temp_data in train_set:
            data_target[0].append(temp_data[portrait1_index][:self.args.dense_fea_n1])
            data_target[1].append(temp_data[portrait2_index][:self.args.dense_fea_n2])
            for d in temp_data[seq1_index]:
                data_target[2].append(d[:self.args.seq_n1])
            for d in temp_data[seq2_index]:
                data_target[3].append(d[:self.args.seq_n2]) 
        
        for i in range(len(self.scaler_list)):
            self.scaler_list[i]=self.scaler_list[i].fit(data_target[i])
        data_target=[]
        del data_target
        gc.collect()

        for temp_data in train_set:
            temp_data[portrait1_index][:self.args.dense_fea_n1]=self.scaler_list[0].transform([temp_data[portrait1_index][:self.args.dense_fea_n1]])[0]
            temp_data[portrait2_index][:self.args.dense_fea_n2]=self.scaler_list[1].transform([temp_data[portrait2_index][:self.args.dense_fea_n2]])[0]
            for d in temp_data[seq1_index]:
                d[:self.args.seq_n1]=self.scaler_list[2].transform([d[:self.args.seq_n1]])[0]
            for d in temp_data[seq2_index]:
                d[:self.args.seq_n2]=self.scaler_list[3].transform([d[:self.args.seq_n2]])[0]
            
            self.node_features1.append(temp_data[portrait1_index])
            self.node_features2.append(temp_data[portrait2_index])
        
        for temp_data in vali_set:
            temp_data[portrait1_index][:self.args.dense_fea_n1]=self.scaler_list[0].transform([temp_data[portrait1_index][:self.args.dense_fea_n1]])[0]
            temp_data[portrait2_index][:self.args.dense_fea_n2]=self.scaler_list[1].transform([temp_data[portrait2_index][:self.args.dense_fea_n2]])[0]
            for d in temp_data[seq1_index]:
                d[:self.args.seq_n1]=self.scaler_list[2].transform([d[:self.args.seq_n1]])[0]
            for d in temp_data[seq2_index]:
                d[:self.args.seq_n2]=self.scaler_list[3].transform([d[:self.args.seq_n2]])[0]
            
            self.node_features1.append(temp_data[portrait1_index])
            self.node_features2.append(temp_data[portrait2_index])

        for temp_data in test_set:
            temp_data[portrait1_index][:self.args.dense_fea_n1]=self.scaler_list[0].transform([temp_data[portrait1_index][:self.args.dense_fea_n1]])[0]
            temp_data[portrait2_index][:self.args.dense_fea_n2]=self.scaler_list[1].transform([temp_data[portrait2_index][:self.args.dense_fea_n2]])[0]
            for d in temp_data[seq1_index]:
                d[:self.args.seq_n1]=self.scaler_list[2].transform([d[:self.args.seq_n1]])[0]
            for d in temp_data[seq2_index]:
                d[:self.args.seq_n2]=self.scaler_list[3].transform([d[:self.args.seq_n2]])[0]
            
            self.node_features1.append(temp_data[portrait1_index])
            self.node_features2.append(temp_data[portrait2_index])
        

        return train_set, vali_set, test_set



class Dataset_MY(Dataset):
    def __init__(self,data,cold_data_prop):
        # init
        self.data=data
        self.len=len(self.data)

        SEED =100
        rng = np.random.RandomState(SEED)
        self.cold_data_indices=rng.choice(list(range(self.len)), size =int(self.len*cold_data_prop), replace=False)

        self.min_num=0
        for i in range(len(data)):
            k=np.array(data[i][3])
            self.min_num=min(k[:,0].min(),self.min_num)
        
        test_cold_data=False
        if test_cold_data:
            self.org_data=self.data
            self.data=[]
            for index in range(self.len):
                if index in self.cold_data_indices:
                    self.data.append(self.org_data[index])

        test_hot_data=False
        if test_hot_data:
            self.org_data=self.data
            self.data=[]
            for index in range(self.len):
                if index not in self.cold_data_indices:
                    self.data.append(self.org_data[index])
            
            

        print('数据集长度：',len(data))

    def __read_data__(self):
        pass
    
    def __getitem__(self, index):
        data=self.data[index]
        if index in self.cold_data_indices:
            temp=torch.tensor(data[3])
            temp[:,0]=self.min_num
            temp[:,1]=-1
            return [torch.tensor(data[0]), torch.tensor(data[1]), torch.tensor(data[2]),
                temp,
                torch.tensor(data[4]), data[5], torch.tensor(data[6]), torch.tensor(data[7])]
        return [torch.tensor(data[0]), torch.tensor(data[1]), torch.tensor(data[2]), torch.tensor(data[3]), torch.tensor(data[4]), data[5], torch.tensor(data[6]), torch.tensor(data[7])]
    
    def __len__(self):
        return len(self.data)

class Dataset_MY_COLD(Dataset):
    def __init__(self,data,cold_data_prop,mode='train'):
        # init
        self.org_data=data
        self.len=len(self.org_data)
        self.mode=mode

        SEED =100
        rng = np.random.RandomState(SEED)
        self.cold_data_indices=rng.choice(list(range(self.len)), size =int(self.len*(cold_data_prop+0.1)), replace=False)

        self.vail_indices=self.cold_data_indices[:int(self.len*0.1)]
        

        self.min_num=0
        for i in range(len(data)):
            k=np.array(data[i][3])
            self.min_num=min(k[:,0].min(),self.min_num)

        self.data=[]
        for index in range(self.len):
            if mode=='train' and  index not in self.cold_data_indices:
                self.data.append(self.org_data[index])
            if index in self.cold_data_indices:
                if mode=='vali' and  index in self.vail_indices:
                    self.data.append(self.org_data[index])
                elif mode=='test':
                    self.data.append(self.org_data[index])


        print('数据集长度：',len(data))

    def __read_data__(self):
        pass
    
    def __getitem__(self, index):
        data=self.data[index]
        if self.mode!='train':
            temp=torch.tensor(data[3])
            temp[:,0]=self.min_num
            #temp[:,1]=-1
            return [torch.tensor(data[0]), torch.tensor(data[1]), torch.tensor(data[2]),
                temp,
                torch.tensor(data[4]), data[5], torch.tensor(data[6]), torch.tensor(data[7])]
        return [torch.tensor(data[0]), torch.tensor(data[1]), torch.tensor(data[2]), torch.tensor(data[3]), torch.tensor(data[4]), data[5], torch.tensor(data[6]), torch.tensor(data[7])]
    
    def __len__(self):
        return len(self.data)




    



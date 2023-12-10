import os
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from utils.tools import EarlyStopping, draw_loss, adjust_learning_rate
from utils.timefeatures import time_features
from utils.metrics import metric
from models.loss import MyLoss , class_trans_pred
from models.model import MixExpert
from data.data_loader import Custormer_Data, Dataset_MY


import multiprocessing as mp
from torch.multiprocessing import Barrier
#from torchinfo import summary
import random

class Exp_LVP(object):
    def __init__(self, args):
        self._init_seed()
        self.args = args
        self.device = self._acquire_device()
        self.train_loader, self.vali_loader, self.test_loader = self._get_data()
        self.model = self._build_model().to(self.device)
        #summary(self.model)
            
        self.model_optim1= self._select_optimizer()
        self.criterion =  self._select_criterion()

    def _init_seed(self):
        SEED =4765
        print("seed",SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(SEED)
    def _build_model(self):
        if self.args.model=='MixExpert':
            model = MixExpert(
                self.args.root_path,
                self.args.dense_fea_n1,
                self.args.sparse_fea_nuniqs1, 
                self.args.dense_fea_n2,
                self.args.sparse_fea_nuniqs2,
                self.args.seq_n1, 
                self.args.seq_n2,
                self.args.item_seq_l,
                self.args.n_out,
                self.args.dropout, 
                self.args.n_heads, 
                self.args.n_hidden, 
                self.args.n_embed_p,
                self.args.seq_encoder_type,
                self.args.embed_type,
                self.args.expert_n,
                self.args.cluster_n)
            return model
       

    
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        train_set, vali_set,test_set = Custormer_Data(self.args).get_data()
      
        train_loader = DataLoader(
                    Dataset_MY(train_set,self.args.cold_data_prop),
                    batch_size=self.args.batch_size,
                    shuffle=True,
                    num_workers=self.args.num_workers,
                    drop_last=False,
                    pin_memory=True,
                    prefetch_factor=1,
                    persistent_workers=True #torch.__version__>='1.8.0' 
                    )
        vali_loader = DataLoader(
                Dataset_MY(vali_set,self.args.cold_data_prop),
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=False,
                pin_memory=True,
                prefetch_factor=1,
                persistent_workers=True #torch.__version__>='1.8.0' 
                )
        
        test_loader = DataLoader(
            Dataset_MY(test_set,self.args.cold_data_prop),
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=True
            )
           
        return train_loader, vali_loader, test_loader

    def _select_optimizer(self,stage='tgt'):
        target_parameters = []
        source_parameters = []
        
        for name, p in self.model.named_parameters():
            if "src" in name:
                source_parameters.append(p)
            elif "tgt" in name:
                target_parameters.append(p)    
               
        model_optim1 = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)#optim.SGD(self.model.parameters(), lr = self.args.learning_rate,momentum=0.9)
        return model_optim1
    
    def _select_inner_optimizer(self,model):
        parameters = []
        for name, p in model.named_parameters():
            if "2" in name or "cross" in name:
                parameters.append(p)

        model_optim = optim.Adam(parameters, lr=self.args.inner_learning_rate)
        return model_optim

    def _refresh_model_parameters(self, new_model, old_model, module='1'):
        for name, p in new_model.named_parameters():
            if module=="1" and "1" in name:
                old_model.state_dict()[name].copy_(p)
            if module=="2" and (~"1" in name):
                old_model.state_dict()[name].copy_(p)
    
    

    def _select_criterion(self):
        #criterion =  nn.MSELoss()
        #criterion =  nn.L1Loss()
        
        criterion = MyLoss(self.model,self.args)
        return criterion

    def vali(self,stage='tgt'):
        self.model.eval()
        vali_loss1 = []
        vali_loss1_detail =[]
        for i, data in enumerate(self.vali_loader):
            loss1, loss1_detail,_,_,_ = self._process_one_batch(data, stage,mode='test')
            vali_loss1.append(loss1.detach())
            vali_loss1_detail.append(loss1_detail.detach())
            
        vali_loss1 = torch.stack(vali_loss1,dim=0).mean(dim=0)
        vali_loss1_detail = torch.stack(vali_loss1_detail,dim=0).mean(dim=0)
        
        self.model.train()
        return vali_loss1,vali_loss1_detail

    def train(self, setting, stage='tgt', load_pre=False,pre_setting=None):
        self._init_seed()
        torch.cuda.empty_cache()
        if load_pre:
            pre_tgt_model_path = self.args.root_path+'/checkpoints/'+pre_setting+'/tgt_model/checkpoint.pth'
            pre_tgt_model_dict=torch.load(pre_tgt_model_path)
            src_pre_setting='MixExpert_normal/pre_GuangZhou_cold_data_prop_0_expert_n_{}'.format(self.args.expert_n)
            if '_wo_le' in pre_setting:
                src_pre_setting=src_pre_setting+'_wo_le'
            pre_src_model_path = self.args.root_path+'/checkpoints/'+src_pre_setting+'/src_model/checkpoint.pth'
            pre_src_model_dict=torch.load(pre_src_model_path)
            model_dict = self.model.state_dict()
        
            pretrained_dict = {}
            for k, _ in model_dict.items():
                if 'tgt_model' in k:
                    pretrained_dict[k] = pre_tgt_model_dict[k]
                if 'src_model' in k:
                    pretrained_dict[k] = pre_src_model_dict[k]
                
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
        
        time_now = time.time()
        
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        

        epoch_train_loss1=[]
        epoch_test_loss1=[]

        # 文件路径创建
        if not os.path.exists(self.args.root_path+'/checkpoints/'+setting):
            os.makedirs(self.args.root_path+'/checkpoints/'+setting)
        if not os.path.exists(self.args.root_path+'/results/'+setting):
            os.makedirs(self.args.root_path+'/results/'+setting)

        
        for epoch in range(self.args.train_epochs):
            self.args.step_count=epoch
            iter_count = 0
            train_loss1 = []
            train_loss1_detail = []
            
            
            self.model.train()
            epoch_time = time.time()
            for i, data in enumerate(self.train_loader):
                iter_count += 1
                loss1, loss1_detail, _, _, _ = self._process_one_batch(data,stage, mode='train')
                
                train_loss1.append(loss1.detach())
                train_loss1_detail.append(loss1_detail.detach())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss1: {2:.7f}".format(i + 1, epoch + 1, loss1.detach()))    
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*len(self.train_loader) - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time),'{:.4f}h'.format(left_time/3600))
                    iter_count = 0
                    time_now = time.time()
                
            
                
                
                

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            
            test_loss1, test_loss1_detail = self.test(setting, stage,load=False)
            train_loss1 = torch.stack(train_loss1,dim=0).mean(dim=0)
            train_loss1_detail = torch.stack(train_loss1_detail,dim=0).mean(dim=0)
            #epoch_train_loss1.append(train_loss1)
            #epoch_test_loss1.append(test_loss1)
            epoch_train_loss1.append(train_loss1_detail.tolist())
            epoch_test_loss1.append(test_loss1_detail.tolist())
            vali_loss1,vali_loss1_detail=self.vali(stage)
            
            print("Epoch: {0}, Steps: {1} | Train loss1: {2:.7f} Vali loss1: {3:.7f} Test loss1: {4:.7f}  ".format(
                    epoch + 1, len(self.train_loader), train_loss1, vali_loss1, test_loss1))


    
            early_stopping(vali_loss1, train_loss1, self.model, self.args.root_path+'/checkpoints/'+setting)
            if early_stopping.early_stop and self.args.step_count>=20:
                print("Early stopping")
                break

            if epoch%1==0:
                draw_loss(epoch_train_loss1, epoch_test_loss1,  self.args.root_path+'/results/'+setting+'/loss1')
        
            
            adjust_learning_rate(self.model_optim1, epoch+1, self.args)
            
            
        best_model_path = self.args.root_path+'/checkpoints/'+setting+'/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        del train_loss1,test_loss1 #内存回收
        
        return self.model

    def test(self, setting, stage='tgt',load=False):
        criterion = self._select_criterion()

        if load:
            best_model_path = self.args.root_path+'/checkpoints/'+setting+'/checkpoint.pth'
            model_dict=torch.load(best_model_path)

            trained_dict = {}
            for k, _ in model_dict.items():
                if 'tgt_model' in k and stage=='tgt':
                    trained_dict[k] = model_dict[k]
                if 'src_model' in k and stage=='src':
                    trained_dict[k] = model_dict[k]
                
            temp_dict=self.model.state_dict()
            temp_dict.update(trained_dict)
            self.model.load_state_dict(temp_dict)

        self.model.eval()
        
        preds = []
        trues = []
        preds2 = []

        test_loss1_detail=[]
        
        test_loss1 = []
        for i, data in enumerate(self.test_loader):
            loss1, loss1_detail, pred, true, pred2 = self._process_one_batch(data, stage, mode='test')
            test_loss1.append(loss1.detach())
            test_loss1_detail.append(loss1_detail)   
            preds.extend(pred.detach().cpu().tolist())
            trues.extend(true.detach().cpu().tolist())
            if pred2 is not None:
                if isinstance(pred2, list):
                    pass
                else:
                    pred2=[pred2.unsqueeze(0)]
                if len(preds2)==0:
                    preds2=[[] for _ in range(len(pred2))]
                
                for i in range(len(pred2)):
                    preds2[i].extend(pred2[i].detach().cpu().tolist())
        test_loss1 = torch.stack(test_loss1,dim=0).mean(dim=0)
        test_loss1_detail = torch.stack(test_loss1_detail,dim=0).mean(dim=0)
        
        
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        for preds2_item in preds2:
            preds2_item = np.array(preds2_item)
            preds2_item = preds2_item.reshape(-1, preds2_item.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        if self.args.loss_type=='Zhou':
            preds=class_trans_pred(preds,preds2)
        mae,nmae,rmse,nrmse,ambe,gini,mape,mspe = metric(preds, trues)
        print('mae:{}, nmae:{}, rmse:{}, nrmse:{}, ambe:{}, gini:{}, mape:{}, mspe:{}'.format(mae,nmae,rmse,nrmse,ambe,gini,mape,mspe))
        
        if load:
            # result save
            result_path = self.args.root_path+'/results/' + setting +'/'
            np.save(result_path+'metrics.npy', np.array([mae,nmae,rmse,nrmse,ambe,gini,mape,mspe]))
            np.save(result_path+'pred.npy', preds)
            np.save(result_path+'true.npy', trues)
            np.save(result_path+'pred2.npy', preds2)
        del pred,true #内存回收    
        return test_loss1,  test_loss1_detail
    

    def _process_one_batch(self, data, stage='tgt', mode='train'):
        data_p1 = data[0].float().to(self.device)
        data_p2 = data[1].float().to(self.device)
        data_stamp = data[2].float().to(self.device)
        data_s1 = data[3].float().to(self.device)
        data_s2_wn = data[4].float().to(self.device)
        data_s2_wi = data[5].float().to(self.device)
        data_y1 = data[-2].float().to(self.device).unsqueeze(-1)
        data_y2 = data[-1].float().to(self.device).unsqueeze(-1)
        if stage in ['src']:
            data_y1=data_y2
        if mode=='train':
            self.model_optim1.zero_grad(set_to_none=True)
            pred1, pred2  = self.model.forward(data_p1,data_p2,data_stamp, data_s1, data_s2_wn, data_s2_wi, stage,mode,self.args.step_count)
            loss1,loss1_detail = self.criterion(pred1, data_y1, pred2, data_y2,stage=stage)
            loss1.backward()
            self.model_optim1.step() 

        elif mode=='test':
            torch.set_grad_enabled(False)
            pred1, pred2 = self.model.forward(data_p1,data_p2,data_stamp, data_s1, data_s2_wn, data_s2_wi, stage,mode,self.args.step_count)
            loss1,loss1_detail = self.criterion(pred1, data_y1, pred2, data_y2,stage=stage)
            
            torch.set_grad_enabled(True)
                
                

        return loss1, loss1_detail, pred1, data_y1[:,0], pred2


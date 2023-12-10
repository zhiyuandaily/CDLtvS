
import torch
print(torch.cuda.is_available())
import argparse
import os

import gc
from utils.tools import Logger
import sys
# sys.path.append()
import time
import numpy as np
import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


print('好滴1111111111111111111111111111111111111111111111111111111111') 
from exp.exp_LVP import Exp_LVP
print(os.getcwd()) 
parser = argparse.ArgumentParser(description='Custormer Life Value Prediction')

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='cross_domain',help='model of experiment, options: [informer, informerstack, informerlight(TBD)]')



parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)

parser.add_argument('--itr', type=int, default=100, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--inner_upadate_steps', type=int, default=3, help='task-level inner update steps')
parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
parser.add_argument('--batch_size', type=int, default=256 , help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--inner_learning_rate', type=float, default=0.001, help='inner optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='optimizer weight decay')
parser.add_argument('--lradj', type=str, default='',help='adjust learning rate')
parser.add_argument('--multi_domain',type=bool, default=False, help='use multiple domains')
parser.add_argument('--pretrain',type=bool, default=False, help='pretrain model in source domain')
parser.add_argument('--pretrain_lradjc',type=float, default=[1,1], help='adjust learning rate coefficient of pretrain module in source domain')
parser.add_argument('--lamdas',type=list, default=[10, 20, 2,1,1], help='loss lamda')
parser.add_argument('--step_count',type=float, default=0, help='train step count')
parser.add_argument('--loss_type',type=str, default='Zhou', help='LVP loss type') #L1Loss, ZILN, SmoothL1Loss, Zhou


parser.add_argument('--dense_fea_n1', type=int, default=0, help='dense features number of the portrait in domain 1')
parser.add_argument('--sparse_fea_nuniqs1', type=list, default=[], help='sparse features nunqiues of the portrait in domain 1')
parser.add_argument('--dense_fea_n2', type=int, default=0, help='dense features number of the portrait in domain 2')
parser.add_argument('--sparse_fea_nuniqs2', type=list, default=[], help='sparse features nunqiues of the portrait in domain 2')
parser.add_argument('--seq_n1', type=int, default=2, help='sequence input size in domain 1')
parser.add_argument('--item_seq_l', type=int, default=30, help='item sequence input length in domain 2')
parser.add_argument('--seq_n2', type=int, default=1, help='sequence input size in domain 2')
parser.add_argument('--n_out', type=int, default=1, help='output dimension in domain 1 and 2')

parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--n_hidden', type=int, default=128, help='dimension of model')
parser.add_argument('--n_embed_p', type=int, default=128, help='dimension of portrait model')
parser.add_argument('--seq_encoder_type', type=str, default='informer', help='sequence encoder model')
parser.add_argument('--embed_type', type=str, default='fixed', help='temporal embedding type')
parser.add_argument('--expert_n', type=int, default=4, help='expert number')
parser.add_argument('--cluster_n', type=int, default=4, help='expert number')

parser.add_argument('--root_path', type=str, default='E:/CDLtvS2023-main', help='root path of the file')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')# GPU*4
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--debug', type=bool, default=False , help='debug program')
parser.add_argument('--use_cache_data', type=bool, default=True , help='use cache data')
parser.add_argument('--data_volume', type=int, default=17, help='data volume')#17
parser.add_argument('--data_normalization',type=bool, default=True, help='data normalization')
parser.add_argument('--data_info',type=str, default='GuangZhou', help='data added info')
parser.add_argument('--cold_data_prop',type=float, default=0.2, help='proportion of cold-start data')
parser.add_argument('--cold_testing',type=bool, default=False, help='test setting whether all cold')

#args = parser.parse_args()
args = parser.parse_args(args=['--model', 'LVP'])
#args = parser.parse_args(args=['--model', 'LSTM'])

#args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False


if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]


args_parser = {
    '0':{'model':'LVP','seq_encoder_type':'tcn','loss_type':'L1Loss','learning_rate':0.001},
    '1':{'model':'LVP','seq_encoder_type':'informer','loss_type':'L1Loss','learning_rate':0.001},
    '2':{'model':'LVP','seq_encoder_type':'wavelet','loss_type':'L1Loss' ,'learning_rate':0.001},
    '3':{'model':'LSTM','seq_encoder_type':None,'loss_type':'L1Loss' ,'learning_rate':0.01},
    '4':{'model':'PTUPCDR','seq_encoder_type':None,'loss_type':'L1Loss' ,'learning_rate':0.001},
    '5':{'model':'TSUR','seq_encoder_type':None,'loss_type':'L1Loss' ,'learning_rate':0.001},
    '6':{'model':'DASL','seq_encoder_type':None,'loss_type':'L1Loss' ,'learning_rate':0.001},
    '7':{'model':'TCN','seq_encoder_type':None,'loss_type':'L1Loss' ,'learning_rate':0.001},
    '8':{'model':'MEInformer','seq_encoder_type':None,'loss_type':'L1Loss' ,'learning_rate':0.001},
    '9':{'model':'MixExpert','seq_encoder_type':'tcn','loss_type':'Zhou' ,'learning_rate':0.001},
    '10':{'model':'Informer','seq_encoder_type':None,'loss_type':'L1Loss','learning_rate':0.001},

}




if not os.path.exists(args.root_path+'/logs'):
    os.makedirs(args.root_path+'/logs')
log_path=args.root_path+'/logs'+'/log_{}.txt'.format(time.strftime("%Y_%m_%d_%H_%M", time.localtime()))
if not args.debug:
    sys.stdout = Logger(log_path)
    sys.stderr =  Logger(log_path)


def init_args(args,Args_SEED):
    data_info = args_parser[str(Args_SEED)]
    args.model=data_info['model']
    args.seq_encoder_type = data_info['seq_encoder_type']
    args.loss_type = data_info['loss_type']
    args.learning_rate = data_info['learning_rate']
    model_suffix='normal'
    return model_suffix

def init_seed(ii,keep_seed=True):
    # torch.backends.cudnn.enable =True
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    if ii==0 or keep_seed:#固定seed
        SEED =4765
    else:
        SEED = random.randint(1,10000)#4853#
    print("seed",SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

cold_data_prop_list=[0,0.2,0.5,0.8]

            
if __name__ == '__main__':
    for ii in range(0,args.itr):
            init_seed(ii)
            args.expert_n=4#2**((ii)%4)#
            args.cold_data_prop=cold_data_prop_list[(ii//4)%4]
            args.cluster_n=1#
            Args_SEED=9
            model_suffix=init_args(args,Args_SEED)
            print('Args in experiment:')
            print(args)
            Exp = Exp_LVP
            # setting record of experiments
            setting = '{}_{}/{}_itr_{}'.format(args.model, model_suffix, args.data_info,ii)
            exp = Exp(args) # set experiments
            if args.model=='MixExpert':
                pre_setting='{}_{}/pre_{}_cold_data_prop_{}'.format(args.model, model_suffix, args.data_info,args.cold_data_prop)
                #pre_setting=setting
                print('>>>>>>>start pre-training target: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(pre_setting+'/tgt_model', load_pre=False,stage='tgt')
                exp.test(pre_setting+'/tgt_model', load=True,stage='tgt')
                print('>>>>>>>start pre-training source: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp = Exp(args)
                exp.train(src_pre_setting+'/src_model', load_pre=False,stage='src')
                exp.test(src_pre_setting+'/src_model', load=True,stage='src')
                print('>>>>>>>start training: {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp = Exp(args)
                exp.train(setting, load_pre=True,stage='meta',pre_setting=pre_setting)
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting)) 
                exp.test(setting, load=True,stage='meta')
            exp=[]
            del exp
            gc.collect()

            

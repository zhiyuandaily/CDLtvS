import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
import random
import os

def draw_loss(train_loss, test_loss, path):
    y_train_loss = np.array(train_loss)
    x_train_loss = range(len(y_train_loss))
    y_test_loss = np.array(test_loss)
    x_test_loss = range(len(y_test_loss))
    
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('epochs')    # x轴标签
    plt.ylabel('loss')     # y轴标签
	# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
	# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss[:,0], linewidth=1, linestyle="solid", label="train loss")
    plt.plot(x_test_loss, y_test_loss[:,0], linewidth=1, linestyle="solid", label="test loss")
    plt.legend()
    plt.title('Loss curve')
    plt.savefig(path+'.png')

    
    for i in range(1,y_train_loss.shape[1]):
        plt.figure()
        # 去除顶部和右边框框
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xlabel('epochs')    # x轴标签
        plt.ylabel('loss')     # y轴标签
        plt.plot(x_train_loss, y_train_loss[:,i], linewidth=1, linestyle="solid", label="train loss")
        plt.plot(x_test_loss, y_test_loss[:,i], linewidth=1, linestyle="solid", label="test loss")
        plt.legend()
        plt.title('Loss curve')
        plt.savefig(path+'_more'+str(i)+'.png')

class Logger(object):
    logfile =""
    def __init__(self, filename=""):
        self.logfile = filename
        self.terminal = sys.stdout
        # self.log = open(filename, "a")
        return

    def write(self, message):
        self.terminal.write(message)
        if self.logfile != "":
            try:
                self.log = open(self.logfile, "a")
                self.log.write(message)
                self.log.close()
            except:
                pass

    def flush(self):
        pass

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj=='type1':
        lr_adjust = {epoch:  0.5}
    elif args.lradj=='type2':
        # 假设初始1e-3
        lr_adjust = {
            2: 0.5, 4: 0.5, 6: 0.5, 8: 0.5, 
            10: 0.5, 15: 0.5, 20: 0.5
        }
    else:
        return
    if epoch in lr_adjust.keys():
        k = lr_adjust[epoch]
        lr = args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr']*k
            lr = param_group['lr']
            
        print('Updating learning rate to {} '.format(lr))

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_vali_score = None
        self.best_train_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, train_loss, model, path):
        vali_score = -val_loss
        train_score= -val_loss
        if self.best_vali_score is None or vali_score >= self.best_vali_score + self.delta:
            self.best_vali_score = vali_score
            self.save_checkpoint(val_loss, model, path)
            
        if self.best_train_score is None or train_score >= self.best_train_score + self.delta:
            self.best_train_score = train_score
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            else:
                self.early_stop = False

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean

def init_seed(seed=4765,keep_seed=True):
    if keep_seed:#固定seed
        SEED =seed
    else:
        SEED = random.randint(1,10000)#4853#
    print("seed",SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

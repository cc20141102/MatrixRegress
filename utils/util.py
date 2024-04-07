

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from utils.dataprocess.read_data import MatrxRegressDataset, npy_file_read
from utils.NNs import AED
from utils.History import History
import matplotlib.pyplot as plt
from utils.Loss import *
from tqdm import tqdm
import numpy as np

# 加载数据集
def load_train_data(config_dict, args):
    data_path = config_dict['data_path'] + '_split'
    train_dataset = MatrxRegressDataset(data_path+'/train', args.debug, mode='train')
    valid_dataset = MatrxRegressDataset(data_path+'/valid', args.debug, mode='valid')
    train_dataloader = DataLoader(train_dataset, args.batch_size, num_workers=4, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, args.batch_size, num_workers=4, shuffle=False)
    return train_dataloader, valid_dataloader 

# 加载数据集
def load_test_data(config_dict, args):
    data_path = config_dict['data_path'] + '_split'
    test_dataset = MatrxRegressDataset(data_path+'/test', args.debug, mode='test')
    test_dataloader = DataLoader(test_dataset, args.batch_size, num_workers=4, shuffle=False)
    return test_dataloader


# 加载模型
def load_model(config_dict, args, mode='train'):
    input_shape = config_dict['input_shape']
    output_shape = config_dict['output_shape']
    model = AED(input_shape, output_shape)
    device_ids = list(map(int, args.device.split(',')))
    model = nn.DataParallel(model.cuda(), device_ids)
    criterion = AEDLoss() # nn.MSELoss() 

    if mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        history = History(4) if args.debug else History(args.epochs) 
        # 热启动
        if args.warm_start == 1:
            # 保存现场数据、画结果图所需的数据
            last_weights = torch.load(args.checkpoint)
            history.load_context(last_weights)

            # 模型读取权重
            model.load_state_dict(history.model_dict)
            optimizer.load_state_dict(history.optim_dict)
        trainner = {'model': model, 'optimizer': optimizer}
        return trainner, criterion, history
    else:
        # 模型读取权重
        weights = torch.load('./weights/{}.pth'.format(args.nntype))
        model.load_state_dict(weights['model_dict'])
        trainner = {'model': model}
        return trainner, criterion

def rescale_sqrt(loss):
    return np.sqrt(loss)

# 训练一次模型
def train_one_epoch(trainner, criterion, train_dataloader):
    # trainner信息分解
    model, optimizer = trainner['model'], trainner['optimizer']
    
    model.train()
    train_loss, train_size = 0.0, 0
    for batch_x, batch_y in train_dataloader:   # tqdm(train_dataloader, desc='train'):
        batch_x, batch_y = npy_file_read(batch_x, batch_y) # 数据文件名形式
        batch_x = batch_x.cuda()   
        batch_y = batch_y.cuda()
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*batch_x.shape[0]
        train_size += batch_x.shape[0]
    train_loss /= train_size
    
    # trainner信息合并
    trainner['model'],  trainner['optimizer'] = model, optimizer
    return rescale_sqrt(train_loss), trainner  # 放缩回原尺度

# 验证一次模型
def valid_one_epoch(trainner, criterion, valid_dataloader):
    model = trainner['model']
    model.eval()
    valid_loss, valid_size = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y in valid_dataloader:   # tqdm(valid_dataloader, desc='valid'):
            batch_x, batch_y = npy_file_read(batch_x, batch_y) # 数据文件名形式
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            valid_loss += loss.item()*batch_x.shape[0]
            valid_size += batch_x.shape[0]
        valid_loss /= valid_size
    return rescale_sqrt(valid_loss)
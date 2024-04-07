# 切片
import json
import argparse
import gc
import traceback
from utils.util import *
import torch
import os
from tqdm import tqdm
import datetime
import numpy as np
import random 
import logging
from torch.utils.tensorboard import SummaryWriter

def main(config_dict, args, info_title):
    # 1. load data
    train_dataloader, valid_dataloader = load_train_data(config_dict, args)
    test_dataloader = load_test_data(config_dict, args)
    logging.info("load data over!")
    
    # 2. load model
    trainner, criterion, history = load_model(config_dict, args)
    logging.info("load model over!")
    
    # 3. 开始训练
    tb = SummaryWriter("logs/tb/{}".format(info_title))
    prev_val_loss, best_val_loss = float('inf'), float('inf')
    if args.warm_start == 1:
        best_val_loss = history.valid_loss[history.start_epoch-1]
    val_no_impv = 0
    for epoch in tqdm(range(history.start_epoch, history.epochs), desc='train loop'):
        
        # 训练阶段
        train_loss, trainner = train_one_epoch(trainner, criterion, train_dataloader)
        gc.collect()
        valid_loss = valid_one_epoch(trainner, criterion, valid_dataloader)
        gc.collect()
      
        # tensorboard添加loss
        tb.add_scalars('{}_loss'.format(info_title), {'train': train_loss, 'valid': valid_loss}, epoch)
        logging.info("epoch: {:d}\t train_loss: {:.4f}\t valid_loss: {:.4f}".format(epoch, train_loss, valid_loss))
        
        # 添加现场到history
        new_context = {'model_dict': trainner['model'].state_dict(),
                       'optim_dict': trainner['optimizer'].state_dict(),
                       'train_loss': train_loss,
                       'valid_loss': valid_loss}
        history.add_context(new_context)

        # 每5次保存现场，计算指标
        if (epoch+1)%5 == 0 or epoch+1 == history.epochs:
            checkpoint_file = './checkpoints/{}_{}.pth'.format(args.nntype, epoch+1)
            history.save_context(checkpoint_file)
            test_loss = valid_one_epoch(trainner, criterion, test_dataloader)
            logging.info('============================================== test_loss: {:.4f}'.format(test_loss)) 
            
        # 保存最佳模型
        if best_val_loss > valid_loss:
            best_val_loss = valid_loss
            weight_file = './weights/{}.pth'.format(args.nntype)
            history.save_context(weight_file)

        # 修改学习率
        if prev_val_loss <= valid_loss:
            val_no_impv += 1
            if val_no_impv == args.half_lr:
                optim_state = trainner['optimizer'].state_dict()
                optim_state['param_groups'][0]['lr'] /= 2.0
                trainner['optimizer'].load_state_dict(optim_state)
            if val_no_impv == args.early_stop:
                logging.info('early stopped...')
                break
        else:
            val_no_impv = 0
        prev_val_loss = valid_loss  
        # 垃圾回收
        gc.collect()
  
    logging.info('trainning over...')   


def get_args():
    parser = argparse.ArgumentParser(description='训练命令行参数')
    
    # 数据集
    parser.add_argument('-debug', type=int, default=0, help='小数据测试程序是否能顺利运行')
    parser.add_argument('-nntype', type=str, default='AED', help='模型名')

    # 训练超参数
    parser.add_argument('-seed', type=int, default=2022, help='随机数种子')
    parser.add_argument('-warm_start', type=int, default=0, help='是否从上一次接着训练')
    parser.add_argument('-checkpoint', type=str, default='./checkpoints/unet_30.pth', help='上一次的模型文件')
    parser.add_argument('-device', type=str, default='0,1', help='cuda')
    parser.add_argument('-batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('-epochs', type=int, default=100, help='epochs')
    parser.add_argument('-lr', type=float, default=2e-3, help='lr')
    parser.add_argument('-half_lr', type=int, default=3, help='连续几次loss没下降, 学习率减半')
    parser.add_argument('-early_stop', type=int, default=10, help='early_stop')
    parser.add_argument('-weight_decay', type=float, default=0.0, help='weight_decay')
    
    return parser.parse_args()


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':  
    # 读取data_path
    with open('./config.json', 'r') as fd:
        config_dict = json.load(fd)
    st_time = datetime.datetime.now()
    args = get_args()
    info_title = args.nntype

    # 设置日志文件
    log_file = "logs/{}.log".format(info_title)
    logging.basicConfig(filename=log_file, level=logging.INFO, handlers=None,
        filemode='a', format='%(asctime)s %(levelname)s>> %(message)s', datefmt='%Y%m%d-%H:%M:%S')
    logging.info('命令行参数: {}'.format(args))
  
    # 随机数种子
    set_random_seed(args.seed)
    
    # 训练
    try:
        main(config_dict, args, info_title)
    except Exception as e:
        logging.error("主程序报错\n{}\n".format(e))
        logging.error(traceback.format_exc())
        exit(0)

    ed_time = datetime.datetime.now()
    logging.info('本次训练时间: {}'.format(ed_time - st_time))





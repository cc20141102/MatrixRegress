
import argparse
import traceback
from utils.util import load_model
from utils.dataprocess.read_data import single_npy_read
import os
import datetime
import logging
import torch
import json
from tqdm import tqdm
import numpy as np

def main(config_dict, args):
    
    # 1. load model
    trainner, criterion = load_model(config_dict, args, mode='test')
    logging.info("load model over!")
    
    # 2. 开始预测
    model = trainner['model']
    model.eval()
    
    if args.mode == 'test':
        result_csv = config_dict['result_csv']
        if os.path.exists(result_csv):
            os.remove(result_csv)
        ## 写入表头
        head_row_list = ['filename', 'average std/point']
        with open(result_csv, 'w') as fd:
            fd.write(",".join(head_row_list)+'\n')
        data_list = os.listdir(config_dict['test_path'])
        data_list = sorted(data_list) 
        num = int(len(data_list) // 2)
        x_list = data_list[num:]
        y_list = data_list[:num]
        with torch.no_grad():
            valid_std = 0.0
            for i in tqdm(range(num), desc='test'):
                x_path = os.path.join(config_dict['test_path'], x_list[i])
                y_path = os.path.join(config_dict['test_path'], y_list[i])
                batch_x = single_npy_read(x_path).cuda()  # [1, 7, 100, 100]
                batch_y = single_npy_read(y_path).cuda()  # [1, 3, 1000, 1000]
                y_pred = model(batch_x)    # [1, 3, 1000, 1000]
                y_pred = torch.round(y_pred)
                loss = criterion(y_pred, batch_y)
                std = np.sqrt(loss.item())
                row = x_list[i] + ',{:.4f}'.format(std)
                with open(result_csv, 'a') as fd:
                    fd.write(row+'\n')
                valid_std += std 
            valid_std /= num
            row = 'tot samples,{:.4f}'.format(valid_std)       
            with open(result_csv, 'a') as fd:
                    fd.write(row+'\n')
    else:
        x_list = os.listdir(config_dict['pred_path'])
        for x_name in tqdm(x_list, desc='predict'):
            x_path = os.path.join(config_dict['pred_path'], x_name)
            batch_x = single_npy_read(x_path).cuda()  # [1, 7, 100, 100]
            y_pred = model(batch_x)    # [1, 3, 1000, 1000]
            y_pred = torch.round(y_pred)
            y_pred = y_pred.squeeze_(0).cpu().detach().numpy() # [3, 1000, 1000]
            
            y_name = x_name.replace('final', '3pass')
            y_path = os.path.join(config_dict['save_path'], y_name)
            np.save(y_path, y_pred)
    
    logging.info('test over...')   

def get_args():
    parser = argparse.ArgumentParser(description='命令行参数')
    
    # 测试类型
    parser.add_argument('-mode', type=str, default='pred', choices=['pred', 'test'], help='pred or test')
    
    # 训练超参数
    parser.add_argument('-nntype', type=str, default='AED', help='模型名')
    parser.add_argument('-device', type=str, default='0', help='cuda')
    
    return parser.parse_args()
 


if __name__ == '__main__':
    
    with open('./config.json', 'r') as fd:
        config_dict = json.load(fd)
    st_time = datetime.datetime.now()
    args = get_args()
    info_title = "{}_{}".format(args.nntype, args.mode)

    # 设置日志文件
    log_file = "logs/{}.log".format(info_title)
    logging.basicConfig(filename=log_file, level=logging.INFO, handlers=None,
        filemode='a', format='%(asctime)s %(levelname)s>> %(message)s', datefmt='%Y%m%d-%H:%M:%S')
    logging.info('命令行参数: {}'.format(args))
    
    # 训练
    try:
        main(config_dict, args)
    except Exception as e:
        logging.error("主程序报错\n{}\n".format(e))
        logging.error(traceback.format_exc())
        exit(0)

    ed_time = datetime.datetime.now()
    logging.info('本次训练时间: {}'.format(ed_time - st_time))





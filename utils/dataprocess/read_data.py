# 创建dataset
# input: [c, h, w] [7, 100, 100]
# label: [3, 1000, 1000]

import torch
from torch.utils.data import Dataset
import os
import json
import numpy as np
from tqdm import tqdm

def npy_file_read(x_data_list, y_data_list):
    x = torch.tensor(np.asarray([np.load(x_data_list[i]) for i in range(len(x_data_list))], dtype=np.float32))
    y = torch.tensor(np.asarray([np.load(y_data_list[i]) for i in range(len(y_data_list))], dtype=np.float32))
    return x, y

def single_npy_read(data_path):
    x = np.load(data_path).astype(np.float32)
    y = torch.tensor(x[np.newaxis, :, :, :])
    return y

class MatrxRegressDataset(Dataset):
    def __init__(self, data_path, debug, mode='train'):
        super().__init__()
        data_list = os.listdir(data_path)
        data_list = sorted(data_list)
        num = int(len(data_list) // 2)
        tot_len = num
        self.input = []
        self.label = []
        if debug:
            tot_len = 230
        for i in tqdm(range(tot_len), desc='{} data'.format(mode)):
            # 读取label.npy文件
            self.label.append(os.path.join(data_path, data_list[i]))
            # 读取x.npy文件
            self.input.append(os.path.join(data_path, data_list[i+num]))
          
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.input[idx], self.label[idx]
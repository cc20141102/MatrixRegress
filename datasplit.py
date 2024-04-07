# 划分数据集
import json
import os
import shutil
from random import shuffle
from tqdm import tqdm
split_rate=[0.7, 0.2, 0.1]
    
def tvt_split(data_path):
    data_list = os.listdir(data_path)
    data_list = sorted(data_list)
    num = int(len(data_list) // 2)
    x_data_list = data_list[num:]   # 3pass < final
    shuffle(x_data_list)
    save_path = data_path + '_split'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)
    file_path = [os.path.join(save_path, x) for x in ['train', 'valid', 'test']]
    for tmp_path in file_path:
        os.mkdir(tmp_path)
    train_len = int(num*split_rate[0])
    valid_len = int(num*(split_rate[0]+split_rate[1]))
    for i in tqdm(range(num), desc='tvt'):
        p = 2
        if i < train_len:
            p = 0
        elif i < valid_len:
            p = 1 
        x_name = x_data_list[i]
        src_path = os.path.join(data_path, x_name)
        dst_path = os.path.join(file_path[p], x_name)
        shutil.copy(src=src_path, dst=dst_path) # copy x
        
        label_name = x_name.replace('final', '3pass')
        src_path = os.path.join(data_path, label_name)
        dst_path = os.path.join(file_path[p], label_name)
        shutil.copy(src=src_path, dst=dst_path) # copy label
        

if __name__ == '__main__':
    with open('./config.json', 'r') as fd:
        data_path = json.load(fd).get('data_path')
    tvt_split(data_path)
    

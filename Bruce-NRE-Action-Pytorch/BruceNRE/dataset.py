
import torch
from torch.utils.data import Dataset
from BruceNRE.utils import load_pkl

class CustomDataset(Dataset):
    def __init__(self, file_path, obj_name):
        self.file = load_pkl(file_path, obj_name)

    def __getitem__(self, item):
        sample = self.file[item]
        return sample#随机采样，在train的时候才会调用

    def __len__(self):
        return len(self.file)
#datasets这是一个pytorch定义的dataset的源码集合。下面是一个自定义Datasets的基本框架，初始化放在__init__()中，
# 其中__getitem__()和__len__()两个方法是必须重写的。__getitem__()返回训练数据.__getitem__负责按索引取出某个数据，并对该数据做预处理。，如图片和label，而__len__()返回数据长度。

def collate_fn(batch):
    batch.sort(key=lambda data: len(data[0]), reverse=True)#128个数据长的在前
    lens = [len(data[0]) for data in batch]#取出一个批次最长的长度进行填充
    max_len = max(lens)

    sent_list = []
    head_pos_list = []
    tail_pos_list = []
    mask_pos_list = []
    relation_list = []

    def _padding(x, max_len):
        return x + [0] * (max_len - len(x))

    for data in batch:
        sent, head_pos, tail_pos, mask_pos, relation = data
        sent_list.append(_padding(sent, max_len))
        head_pos_list.append(_padding(head_pos, max_len))
        tail_pos_list.append(_padding(tail_pos, max_len))
        mask_pos_list.append(_padding(mask_pos, max_len))
        relation_list.append(relation)

    return torch.tensor(sent_list), torch.tensor(head_pos_list), torch.tensor(tail_pos_list), torch.tensor(mask_pos_list), torch.tensor(relation_list)
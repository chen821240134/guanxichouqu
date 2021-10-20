

import torch
import torch.nn as nn
import time
from BruceNRE.utils import ensure_dir

class BasicModule(nn.Module):
    """
    封装nn.Module,提供save和load方法
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    """
    加载指定路径的模型
    """
    def load(self, path):
        self.load_state_dict(torch.load(path))#torch.load_state_dict()函数就是用于将预训练的参数权重加载到新的模型之中

    """
    保存模型
    """
    def save(self, epoch=0, name=None):
        prefix = 'checkpoints/'
        ensure_dir(prefix)#确认前缀存在 自定义函数确认文件夹存在，不在就创建
        if name is None:
            name = prefix + self.model_name + "_" + f'epoch{epoch}_'
            name = time.strftime(name + '%m%d_%H_%M_%S.pth')
        else:
            name = prefix + name + '_' + self.model_name + "_" + f'epoch{epoch}_'
            name = time.strftime(name + '%m%d_%H_%M_%S.pth')#时分秒

        torch.save(self.state_dict(), name)
        return name

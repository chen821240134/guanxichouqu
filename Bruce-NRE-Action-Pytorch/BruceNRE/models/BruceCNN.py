
import torch
import torch.nn as nn
import torch.nn.functional as F
from BruceNRE.models import BasicModule, Embedding

class BruceCNN(BasicModule):#先跳转至BASIC
    def __init__(self, vocab_size, config):#因为表是根据统计得出，现在还无法确定
        super(BruceCNN, self).__init__()
        self.model_name = 'BruceCNN'
        self.out_channels = config.out_channels#100
        self.kernel_size = config.kernel_size#[3,5]
        self.word_dim = config.word_dim#300
        self.pos_size = config.pos_size#102
        self.pos_dim = config.pos_dim#10
        self.hidden_size = config.hidden_size#100
        self.dropout = config.dropout#0.5
        self.out_dim = config.relation_type#10

        if isinstance(self.kernel_size, int):#isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。  判断size是否是整型
            self.kernel_size = [self.kernel_size]
        for k in self.kernel_size:
            assert k % 2 == 1, 'k 必须是奇数'

        self.embedding = Embedding(vocab_size, self.word_dim, self.pos_size, self.pos_dim)

        self.input_dim = self.word_dim + self.pos_dim * 2#320

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.input_dim,
                      out_channels=self.out_channels,
                      kernel_size=k,
                      padding=k//2,
                      bias=None
                      ) for k in self.kernel_size
        ])
# #ModuleList(
#   (0): Conv1d(320, 100, kernel_size=(3,), stride=(1,), padding=(1,), bias=False)
#   (1): Conv1d(320, 100, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
# )
        self.conv_dim = len(self.kernel_size) * self.out_channels#200
        self.fc1 = nn.Linear(self.conv_dim, self.hidden_size)#200到100
        self.dropout = nn.Dropout(self.dropout)#
        self.fc2 = nn.Linear(self.hidden_size, self.out_dim)

    def forward(self, input):
        """
        :param self:
        :param input: word_ids, headpos, tailpos, mask
        :return:
        """
        *x, mask = input#*x代表了前三个输出word_ids, headpos, tailpos   mask是预留，bert用得到

        x = self.embedding(x)#128 124 320
        x = torch.transpose(x, 1, 2)#128 320 124 124是一个batch的最长长度，会动态改变 batch emb_dim seqlen

        x = [F.leaky_relu(conv(x)) for conv in self.convs]#返回两个list128 100 124  Conv1d(320, 100, kernel_size=(5,), stride=(1,), padding=(2,), bias=False)
        x = torch.cat(x, dim=1)#128 200 124
        s_len = x.size(-1)#124
        x = F.max_pool1d(x, s_len)#128 200 1#创建一个池化大小为s_len的池化层
        x = x.squeeze(-1)#128 200

        x = self.dropout(x)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x



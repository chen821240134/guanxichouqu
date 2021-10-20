
import torch
import torch.nn as nn



class Embedding(nn.Module):
    def __init__(self, vocab_size, word_dim, pos_size, pos_dim):
        super(Embedding, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, word_dim, padding_idx=0)#6618 200#填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。
        self.head_pos_embed = nn.Embedding(pos_size, pos_dim, padding_idx=0)#102 10
        self.tail_pos_embed = nn.Embedding(pos_size, pos_dim, padding_idx=0)#102 10


    def forward(self, x):
        words, head_pos, tail_pos = x#大小都是128 124
        word_embed = self.word_embed(words)#128 124 300
        head_embed = self.head_pos_embed(head_pos)#128 124 10
        tail_embed = self.tail_pos_embed(tail_pos)#128 124 10
        feature_embed = torch.cat([word_embed, head_embed, tail_embed], dim=-1)#128 124 320

        return feature_embed


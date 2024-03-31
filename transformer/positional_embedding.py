import math

import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,max_seq_len,d_model):
        super(PositionalEncoding, self).__init__()
        pe =torch.zeros(max_seq_len,d_model)

        # 转化为列向量 max_len * 1
        position = torch.arange(0,max_seq_len,dtype=torch.float).unsqueeze(1)

        # 线性变换矩阵 1*d_model//2
        div_term = torch.exp(torch.arange(0,d_model,2,dtype=torch.float)*-(math.log(10000)/d_model))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        # 此时pe维度为 max_seq_len * d_model 为了与输入相加,还需要扩展维度
        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):
        # x是输入的word embedding batch_size * seq_len * d_model
        # pe是计算好的位置编码 1 * max_seq_len * d_model
        # print(self.pe.size())
        # print(x.size(1))
        return x + self.pe[:,:x.size(1)]

if __name__ == '__main__':
    batch_size = 2
    max_seq_len = 10
    d_model = 64
    embedding = PositionalEncoding(max_seq_len,d_model)
    position_embedding = embedding(torch.randn(batch_size,max_seq_len,d_model))




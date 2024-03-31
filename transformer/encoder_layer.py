import torch
import torch.nn as nn
from multi_head_self_attention import MultiHeadSelfAttention
from feed_forward import FeedForward
class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        # dropout率用于回归训练
        super(EncoderLayer,self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model,num_heads)
        self.feed_forward = FeedForward(d_model,d_ff)

        # LayerNormalization的参数必须等于tensor最后一个维度的大小
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 定义dropout层,用来防止过拟合,其原理是在训练时随机地将一些量设置为0
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        # x的维度是 batch_size * seq_len * d_model
        attn_output = self.self_attn(x,x,x,mask)

        x = self.norm1(x+self.dropout(attn_output))
        ff_out = self.feed_forward(x)

        x = self.norm2(x+self.dropout(ff_out))

        return x




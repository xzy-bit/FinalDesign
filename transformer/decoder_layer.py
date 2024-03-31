import torch
import torch.nn as nn
from feed_forward import FeedForward
from multi_head_self_attention import MultiHeadSelfAttention

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeadSelfAttention(d_model,num_heads)
        self.cross_attn = MultiHeadSelfAttention(d_model,num_heads)
        self.feedforward = FeedForward(d_model,d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,enc_output,src_mask,tgt_mask):
        # src_mask 用来忽略encoder_output中的某些部分
        # tgt_mask 用来忽略decoder_input中的某些部分

        self_attn = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x+self.dropout(self_attn))

        cross_attn = self.cross_attn(enc_output,enc_output,x,src_mask)
        x = self.norm2(x+self.dropout(cross_attn))

        ff_out = self.feedforward(x)

        x = self.norm3(x+self.dropout(ff_out))

        return x



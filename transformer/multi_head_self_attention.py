import math
import copy
import torch
import torch.nn as nn
import torch.utils.data as data
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model%num_heads==0,"d_model must be divisible by num_heads"

        # input_size = batch_size * seq_len * d_model
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model//num_heads

        self.W_k = nn.Linear(d_model,d_model)
        self.W_q = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def scaled_dot_product_attention(self,Q,K,V,mask=None):
        # print(Q.size())
        # (Q,K,V)_size = batch_size * num_heads * seq_len * d_k

        # attn_values_size = batch_size * num_heads * seq_len * seq_len
        attn_values = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d_k)

        if mask is not None:
            attn_values = attn_values.masked_fill(mask==0,-1e9)

        # obtain attention probabilities
        attn_probs = torch.softmax(attn_values,dim=-1)

        # obtain final output
        attn_scores = torch.matmul(attn_probs,V)

        # attn_scores_size = batch_size * num_heads * seq_len * d_k
        # print(attn_scores.size())
        return attn_scores

    def split_heads(self,x):
        batch_size,seq_len,d_model = x.size()
        return x.view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)

    def combine_heads(self,x):
        # attn_scores_size = batch_size * num_heads * seq_len * d_k
        batch_size,_,seq_len,d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)

    def forward(self,Q,K,V,mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_q(K))
        V = self.split_heads(self.W_q(V))
        attn_output = self.scaled_dot_product_attention(Q,K,V,mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

if __name__ == '__main__':
    batch_size = 10
    seq_len = 4
    d_model = 64
    num_heads = 16
    triu = (1-torch.triu(torch.ones(1,seq_len,seq_len),diagonal=1)).bool()
    print(triu)

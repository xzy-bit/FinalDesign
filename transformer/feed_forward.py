import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff):
        super(FeedForward,self).__init__()
        self.fc1 = nn.Linear(d_model,d_ff)
        self.fc2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        #输入x的维度为 batch_size * seq_len * d_model
        return self.fc2(self.relu(self.fc1(x)))
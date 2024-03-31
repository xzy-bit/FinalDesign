import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer

if __name__ == '__main__':
    src_vocab_size = 100
    tgt_vocab_size = 100
    d_model = 64
    num_heads = 8
    num_layers = 6
    d_ff = 256
    max_seq_len = 20
    dropout = 0.1

    transformer = Transformer(src_vocab_size,tgt_vocab_size,d_model,num_heads, num_layers, d_ff, max_seq_len, dropout)

    batch_size = 4
    src_data = torch.randint(1,src_vocab_size,(batch_size,max_seq_len))
    tgt_data = torch.randint(1,tgt_vocab_size,(batch_size,max_seq_len+1))

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(),lr=0.0001,betas=(0.9,0.98),eps=1e-9)

    transformer.train()

    for epoch in range(15):
        optimizer.zero_grad()
        output = transformer(src_data,tgt_data[:,:-1])
        loss = criterion(output.contiguous().view(-1,tgt_vocab_size),tgt_data[:,1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch:{epoch+1},Loss: {loss.item()}")

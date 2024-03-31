import torch
import torch.nn as nn
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer
from positional_embedding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self,src_vocab_size,tgt_vocab_size,d_model,num_heads,num_layers,d_ff,max_seq_len,dropout):
        super(Transformer,self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size,d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size,d_model)
        self.positional_encoding = PositionalEncoding(max_seq_len,d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model,num_heads,d_ff,dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model,tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self,src, tgt):
        # src_size = batch_size * max_seq_len
        # attn_values_size = batch_size * num_heads * seq_len * seq_len
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        diagonal = (1-torch.triu(torch.ones(1,seq_len,seq_len),diagonal=1)).bool()
        tgt_mask = tgt_mask & diagonal
        return src_mask,tgt_mask

    def forward(self,src,tgt):
        src_mask,tgt_mask = self.generate_mask(src,tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output,src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output,enc_output,src_mask,tgt_mask)

        output = self.fc(dec_output)
        return output

if __name__ == '__main__':
    batch_size = 4
    max_seq_len = 20
    input = torch.randn(batch_size,max_seq_len)
    print(input.unsqueeze(0).unsqueeze(1).size())




import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) 
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)  

    def forward(self, src, tgt, src_mask, tgt_mask, target=None):
        src = self.embedding(src) 
        tgt = self.embedding(tgt) 

        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        logits = self.lm_head(dec_output)  

        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            return logits, loss  

        return logits

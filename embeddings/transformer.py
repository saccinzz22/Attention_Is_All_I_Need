import torch
import torch.nn as nn
from embedding.positional import PositionalEncoding
from embedding.token import TokenEmbedding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.positional_encoding(self.token_embedding(x)))

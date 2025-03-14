class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.attention(x, x, x, mask)
        x = self.norm1(attn + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


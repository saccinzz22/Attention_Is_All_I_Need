class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)
        self.norm3 = LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, forward_expansion)
        self.dropout = nn.Dropout(dropout)
        self.encoder_attention = MultiHeadAttention(embed_size, heads)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.self_attention(x, x, x, trg_mask)
        query = self.norm1(attention + x)
        attention = self.encoder_attention(query, key, value, src_mask)
        x = self.norm2(attention + query)
        forward = self.feed_forward(x)
        out = self.norm3(forward + x)
        return out

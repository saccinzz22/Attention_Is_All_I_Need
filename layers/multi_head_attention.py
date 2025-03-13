class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.d_k ** 0.5
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return self.w_o(output)

import os
import torch
import torch.optim as optim
import tiktoken
from model import Transformer

tokenizer = tiktoken.get_encoding("gpt2")

device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

data_path =" /Users/sachin/Documents/input/txt"
with open(data_path, "r") as f:
    text = f.read()

tokens = tokenizer.encode(text[:1000])

B, T = 4, 32
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
vocab_size = tokenizer.n_vocab
dropout = 0.1

buf = torch.tensor(tokens[:B*T + 1], dtype=torch.long, device=device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

src_mask = tgt_mask = None  

model = Transformer(num_layers, d_model, num_heads, d_ff, vocab_size, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


epochs = 3
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    logits, loss = model(x, x, src_mask, tgt_mask, target=y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

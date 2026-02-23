import torch.nn as nn
import torch
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, head_size, n_emb, block_size):
        super().__init__()
        self.key = nn.Linear(32, 32, bias=False)
        self.query = nn.Linear(32, 32, bias=False)
        self.value = nn.Linear(32, 32, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(8, 8)))
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, HEAD_SIZE)
        q = self.query(x) # (B, T, HEAD_SIZE)

        # we compute the attention scores ("affinities")
        w = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, HEAD_SIZE) @ (B, HEAD_SIZE, T) / HEAD_SIZE SQUARE ROOT
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T ,T)
        w = F.softmax(w, dim=-1) # (B, T, T)

        # perform the weighted aggregation of the values

        v = self.value(x)
        out = w @ v
        return out


        
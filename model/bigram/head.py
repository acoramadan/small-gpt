"""
head.py - Self-Attention Head & Multi-Head Attention

This module implements the core attention mechanism of the Transformer:
  - Head:               A single causal (masked) self-attention head.
  - MultiHeadAttention: Multiple attention heads running in parallel,
                        whose outputs are concatenated and linearly projected.
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class Head(nn.Module):
    """A single head of causal (masked) self-attention.

    Each head independently learns key, query, and value projections.
    A lower-triangular mask ensures that token at position *t* can only
    attend to tokens at positions <= *t* (autoregressive property).

    Args:
        head_size (int):  Dimensionality of key/query/value projections.
        n_emb (int):      Input embedding dimension.
        block_size (int): Maximum sequence length (used to build the causal mask).
        dropout (float):  Dropout probability applied to attention weights.
    """

    def __init__(self, head_size, n_emb, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """Compute masked self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Attention output of shape (B, T, head_size).
        """
        B, T, C = x.shape
        k = self.key(x)    # (B, T, HEAD_SIZE)
        q = self.query(x)  # (B, T, HEAD_SIZE)

        # Compute attention scores ("affinities"), scaled by sqrt(head_size)
        w = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # causal mask
        w = F.softmax(w, dim=-1)  # (B, T, T)
        w = self.dropout(w)

        # Weighted aggregation of the values
        v = self.value(x)
        out = w @ v

        return out


class MultiHeadAttention(nn.Module):
    """Multiple self-attention heads running in parallel.

    Each head produces an output of dimension `head_size`. The outputs of all
    heads are concatenated along the channel dimension and projected back to
    `n_emb` dimensions via a linear layer, followed by dropout.

    Args:
        num_heads (int):  Number of parallel attention heads.
        head_size (int):  Dimensionality of each individual head.
        n_emb (int):      Input/output embedding dimension.
        block_size (int): Maximum sequence length (passed to each Head).
        dropout (float):  Dropout probability.
    """

    def __init__(self, num_heads, head_size, n_emb, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_emb, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Run all heads in parallel, concatenate, and project.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, n_emb).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, n_emb).
        """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
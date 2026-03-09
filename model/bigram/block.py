"""
block.py - Transformer Block

This module defines a single Transformer block that combines:
  1. A multi-head self-attention sub-layer (communication between tokens).
  2. A position-wise feed-forward sub-layer (computation per token).

Both sub-layers use pre-norm (LayerNorm applied *before* the sub-layer)
and residual connections (output = input + sub_layer(norm(input))).
"""

import torch.nn as nn
from .head import MultiHeadAttention
from .forward import FeedForward


class Block(nn.Module):
    """A single Transformer block with pre-norm residual connections.

    Data flow:
        x  ->  LayerNorm  ->  MultiHeadAttention  ->  (+x residual)  ->
           ->  LayerNorm  ->  FeedForward          ->  (+x residual)  -> out

    Args:
        n_embd (int):     Embedding dimension.
        n_head (int):     Number of attention heads.
        block_size (int): Maximum sequence length.
        dropout (float):  Dropout probability.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            head_size=head_size,
            n_emb=n_embd,
            num_heads=n_head,
            block_size=block_size,
            dropout=dropout
        )
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Apply self-attention and feed-forward with residual connections.

        Args:
            x (torch.Tensor): Input of shape (B, T, n_embd).

        Returns:
            torch.Tensor: Output of shape (B, T, n_embd).
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
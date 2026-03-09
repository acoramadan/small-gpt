"""
bigram.py - Bigram (GPT-style) Language Model

This module defines the top-level Transformer-based language model. Despite the
name "Bigram", the model has evolved beyond a simple bigram lookup table into a
full decoder-only Transformer with:
  - Token + positional embeddings
  - N stacked Transformer blocks (self-attention + feed-forward)
  - A final LayerNorm and linear projection to vocabulary logits
  - Autoregressive text generation via nucleus/multinomial sampling
"""

import torch.nn as nn
import torch
from torch.nn import functional as F
from .head import MultiHeadAttention
from .forward import FeedForward
from .block import Block


class BigramLanguageModel(nn.Module):
    """Decoder-only Transformer language model.

    Architecture overview:
        1. Token embedding table    (vocab_size -> n_embd)
        2. Positional embedding table (block_size -> n_embd)
        3. N x Transformer Block    (self-attention + feed-forward)
        4. Final LayerNorm
        5. Linear head              (n_embd -> vocab_size)

    Args:
        vocab_size (int): Size of the character/token vocabulary.
        n_embd (int):     Embedding dimension.
        block_size (int): Maximum context window length.
        device (str):     Device string ('cuda' or 'cpu').
        dropout (float):  Dropout probability.
        n_layer (int):    Number of Transformer blocks.
        n_head (int):     Number of attention heads per block.
    """

    def __init__(self, vocab_size, n_embd, block_size, device, dropout, n_layer, n_head):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_embd)
        self.position_emb_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
        self.device = device

    def forward(self, idx, targets=None):
        """Forward pass: compute logits and optionally the cross-entropy loss.

        Args:
            idx (torch.Tensor):     Input token indices of shape (B, T).
            targets (torch.Tensor):  Target token indices of shape (B, T), or None.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
                logits -- Raw predictions of shape (B, T, vocab_size) (or (B*T, vocab_size) when targets provided).
                loss   -- Scalar cross-entropy loss, or None if targets is None.
        """
        B, T = idx.shape
        tok_emb = self.token_emb_table(idx)                                   # (B, T, C)
        pos_emb = self.position_emb_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)   # reshape to 2-D for cross_entropy
            targets = targets.view(-1)        # flatten to 1-D (B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressively generate new tokens.

        Given a conditioning context `idx`, the model repeatedly:
          1. Crops the context to the last `block_size` tokens.
          2. Runs a forward pass to get next-token logits.
          3. Applies softmax to obtain a probability distribution.
          4. Samples the next token via multinomial sampling.
          5. Appends the sampled token to the running sequence.

        Args:
            idx (torch.Tensor):      Starting context of shape (B, T).
            max_new_tokens (int):    Number of new tokens to generate.

        Returns:
            torch.Tensor: Extended sequence of shape (B, T + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                   # (B, C) — last time step
            probs = F.softmax(logits, dim=-1)            # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)      # (B, T+1)
        return idx
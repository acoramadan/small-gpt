"""
forward.py - Position-wise Feed-Forward Network

This module implements the feed-forward sub-layer used inside each
Transformer block. It applies two linear transformations with a ReLU
activation in between, expanding the hidden dimension by 4x before
projecting back down.
"""

import torch.nn as nn


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Architecture:
        Linear(n_embd -> 4*n_embd) -> ReLU -> Linear(4*n_embd -> n_embd) -> Dropout

    The inner dimension expansion (4x) gives the network more capacity to
    learn complex token-level transformations.

    Args:
        n_embd (int):   Input and output embedding dimension.
        dropout (float): Dropout probability applied after the second linear layer.
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """Apply the feed-forward network to every position independently.

        Args:
            x (torch.Tensor): Input of shape (B, T, n_embd).

        Returns:
            torch.Tensor: Output of shape (B, T, n_embd).
        """
        return self.net(x)
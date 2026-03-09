"""
util.py - Utility Module for NanoGPT

This module contains all hyperparameter configurations and helper functions
used during training and evaluation of the Transformer-based language model.

Hyperparameters:
    BATCH_SIZE (int):       Number of independent sequences processed in parallel (default: 64).
    BLOCK_SIZE (int):       Maximum context length for predictions (default: 256).
    MAX_ITERS (int):        Total number of training iterations (default: 5000).
    EVAL_INTERVAL (int):    How often to evaluate train/val loss (default: every 500 steps).
    EVAL_ITERS (int):       Number of batches used to estimate loss (default: 200).
    N_EMBD (int):           Embedding dimension size (default: 384, i.e. 64 per head x 6 heads).
    DROPOUT (float):        Dropout rate for regularization (default: 0.2).
    N_LAYER (int):          Number of Transformer blocks stacked sequentially (default: 6).
    N_HEAD (int):           Number of self-attention heads per block (default: 6).
    LR (float):             Learning rate for the AdamW optimizer (default: 3e-4).
    DEVICE (str):           Automatically set to 'cuda' if a GPU is available, else 'cpu'.
"""

import torch

BATCH_SIZE = 64
BLOCK_SIZE = 256
MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 200
N_EMBD = 384 # 384 / 6 = 64 dim for each head
DROPOUT = 0.2
N_LAYER = 6
N_HEAD = 6
LR = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(data, context_length=BLOCK_SIZE, batch_size=BATCH_SIZE):
    """Sample a random mini-batch of input-target pairs from the dataset.

    Randomly selects `batch_size` starting indices, then slices sequences of
    length `context_length` for inputs (x) and the corresponding shifted-by-one
    sequences for targets (y).

    Args:
        data (torch.Tensor):     1-D tensor of encoded token indices.
        context_length (int):    Length of each input sequence (default: BLOCK_SIZE).
        batch_size (int):        Number of sequences per batch (default: BATCH_SIZE).

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            x -- input sequences of shape (batch_size, context_length).
            y -- target sequences of shape (batch_size, context_length).
    """
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y


@torch.no_grad()
def estimate_loss(data_train, data_val, model, eval_iters=EVAL_ITERS):
    """Estimate the average loss on the training and validation splits.

    Puts the model in evaluation mode, computes the mean loss over
    `eval_iters` random batches for each split, then restores training mode.

    Args:
        data_train (torch.Tensor): 1-D tensor of the training data.
        data_val (torch.Tensor):   1-D tensor of the validation data.
        model (nn.Module):         The language model to evaluate.
        eval_iters (int):          Number of batches to average over (default: EVAL_ITERS).

    Returns:
        dict[str, torch.Tensor]: Dictionary with keys 'train' and 'val',
            each mapping to the mean loss (scalar tensor) for that split.
    """
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):

            if split == 'train':
                X, Y = get_batch(data_train)
            else:
                X, Y = get_batch(data_val)

            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
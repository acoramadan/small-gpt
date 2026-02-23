import torch.nn as nn
import torch
from torch.nn import functional as F
from .head import Head
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, device):
        super().__init__()
        self.token_emb_table = nn.Embedding(vocab_size, n_embd) # setiap token langsung membaca dari logits untuk token berikutnya dari lookup table
        self.position_emb_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.sa_head = Head(n_emb=n_embd, head_size=n_embd, block_size=block_size) # self attetion
        self.block_size = block_size
        self.device = device

    def forward(self, idx, targets=None):
        B, T  = idx.shape
        tok_emb = self.token_emb_table(idx) # (B, T, C) Batch, Time, Channel
        pos_emb = self.position_emb_table(torch.arange(T, device=self.device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_head(x)
        logits = self.lm_head(x) # (B, T, C) Batch, Time, Vocab size

        if targets is None: 
            loss = None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # Menyesuaikan dengan argument yang dibutuhkan oleh cross entropy yaitu 2d vector
            targets = targets.view(-1) # mengubah menjadi 1d vector atau B*T
            loss = F.cross_entropy(logits, targets)
 
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx adalah Batch, Time (B, T) indeks array dari konteks sekarang 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            # Mengambil prediksi
            logits, _ = self(idx_cond)
            # fokus hanya untuk step terakhir 
            logits = logits[:, -1, :] # berubah menjadi (B, C) dari (B, T)
            # Gunakan activation function softmax untuk mendapatkan probabilitas
            probs = F.softmax(logits, dim=-1) # Dim (B, C)
            # sample dari hasil softmax
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx  
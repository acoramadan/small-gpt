import torch.nn as nn
import torch
from torch.nn import functional as F

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # setiap token langsung membaca dari logits untuk token berikutnya dari lookup table
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C) Batch, Time (Vocab_size), Channel
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
            # Mengambil prediksi
            logits, _ = self(idx)
            # fokus hanya untuk step terakhir 
            logits = logits[:, -1, :] # berubah menjadi (B, C) dari (B, T)
            # Gunakan activation function softmax untuk mendapatkan probabilitas
            probs = F.softmax(logits, dim=-1) # Dim (B, C)
            # sample dari hasil softmax
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
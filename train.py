import util.util as u
import torch
from model.bigram.bigram import BigramLanguageModel

with open('data/input.txt', 'r', encoding='utf-8') as f:
    data = f.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(data), dtype=torch.long)
n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

model = BigramLanguageModel(vocab_size)
m = model.to(u.DEVICE)

optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

for iter in range(u.MAX_ITERS):
    if iter % u.EVAL_INTERVAL == 0:
        losses = u.estimate_loss(train_data, val_data, model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = u.get_batch(train_data)
    logits, loss = model(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

context = torch.zeros((1, 1), dtype=torch.long, device=u.DEVICE)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
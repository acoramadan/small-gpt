import torch

BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 10000
EVAL_INTERVAL = 300
EVAL_ITERS = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_batch(data, context_length=BLOCK_SIZE, batch_size=BATCH_SIZE ):
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y

@torch.no_grad()
def estimate_loss(data_train, data_val, model, eval_iters=EVAL_ITERS,):
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):

            if split == 'train':
                X, Y = get_batch(data_train)
            else:
                X, Y = get_batch(data_val)

            _, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
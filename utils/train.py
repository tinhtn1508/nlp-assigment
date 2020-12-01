import torch

def batchify(data, batch_size, device='cpu'):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, batch_size * nbatch)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

def get_batch(source, id, sequence_len):
    data = source[id : id + sequence_len]
    target = source[id + 1 : id + 1 + sequence_len].view(-1)
    return data, target

def repackage_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)
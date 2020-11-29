import torch
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)
    sofmax_values = torch.nn.functional.softmax(output, dim=1)
    sofmax_values = sofmax_values.cpu().numpy()
    target = target.cpu().numpy()
    accs = sofmax_values[target == 1]
    return sum(accs) / len(accs)

def cross_entropy(input, target, size_average=True):
    logsoftmax = torch.nn.functional.log_softmax
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input, dim=1), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input, dim=1), dim=1))
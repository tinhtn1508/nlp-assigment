import utils
import model
import torch
import torch.nn as nn
import numpy as np
import time

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def adjust_learning_rate(optimizer, epoch, decay_epoch):
    lr = 0.001
    for epc in decay_epoch:
        if epoch >= epc:
            lr = lr * 0.1
        else:
            break
    print()
    print('Learning Rate:', lr)
    print()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    batch_size = target.size(0)
    attr_num = target.size(1)

    output = torch.sigmoid(output).cpu().numpy()
    output = np.where(output > 0.5, 1, 0)
    pred = torch.from_numpy(output).long()
    target = target.cpu().long()
    correct = pred.eq(target)
    correct = correct.numpy()

    res = []
    for k in range(attr_num):
        res.append(1.0*sum(correct[:,k]) / batch_size)
    return sum(res) / attr_num

def main():
    train_dataset = utils.SequenceDataset('data/nlp_dataset.txt', 'data/nlp_character_vocab.pkl')
    vocab_size = len(train_dataset.vocab_info)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True, num_workers=1, pin_memory=True)
    print(train_loader)
    _model = model.SimpleLSTM(embedding_dim=50, hidden_dim=512, vocab_size=vocab_size, tagset_size=vocab_size)
    # _model = torch.nn.DataParallel(_model).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(_model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.0005)

    
    
    for epoch in range(0, 20):
        # adjust_learning_rate(optimizer, epoch, .decay_epocargsh)
        train(train_loader, _model, criterion, optimizer, epoch)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    end = time.time()
    for i, _ in enumerate(train_loader):
        print(i)
        # input, target = _
        # # target = target.cuda(non_blocking=True)
        # # input = input.cuda(non_blocking=True)
        # output = model(input)

        # bs = target.size(0)
        # loss = criterion.forward(torch.sigmoid(output), target, epoch)

        # # measure accuracy and record loss
        # accu = accuracy(output.data, target)
        # losses.update(loss.data, bs)
        # top1.update(accu, bs)

        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # # measure elapsed time
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if i % 10 == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
        #           epoch, i, len(train_loader), batch_time=batch_time,
        #           loss=losses, top1=top1))

if __name__ == "__main__":
    main()
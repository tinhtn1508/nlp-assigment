import utils
import model
import torch
import torch.nn as nn
import time
import os

def save_checkpoint(state, epoch, prefix, filename='.pth'):
    directory = "./"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if prefix == '':
        filename = directory + str(epoch) + filename
    else:
        filename = directory + prefix + '_' + str(epoch) + filename
    torch.save(state, filename)

def main():
    train_dataset = utils.SequenceDataset('data/nlp_train_dataset.txt', 'data/nlp_character_vocab.pkl')
    val_dataset = utils.SequenceDataset('data/nlp_val_dataset.txt', 'data/nlp_character_vocab.pkl')
    vocab_size = len(train_dataset.vocab_info)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2000, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=500, shuffle=True, num_workers=2, pin_memory=True)
    _model = model.SimpleLSTM(embedding_dim=512, hidden_dim=50, vocab_size=vocab_size, tagset_size=vocab_size)
    _model = torch.nn.DataParallel(_model).cuda()
    criterion = utils.cross_entropy
    optimizer = torch.optim.SGD(_model.parameters(), lr=0.1)

    best_accu = 0
    for epoch in range(0, 100):
        print("============"*8)
        train(train_loader, _model, criterion, optimizer, epoch)
        accu = validate(val_loader, _model, criterion, epoch)
        best_accu = max(accu, best_accu)

        if epoch in [30]:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': _model.state_dict(),
                'best_accu': best_accu,
            }, epoch+1, "")

def train(train_loader, _model, criterion, optimizer, epoch):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    _model.train()

    end = time.time()
    for i, _ in enumerate(train_loader):
        input, target = _
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        output = _model(input)
        bs = target.size(0)
        loss = criterion(output, target)

        # measure accuracy and record loss
        accu = utils.accuracy(output.data, target)
        losses.update(loss.data, bs)
        top1.update(accu, bs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  loss=losses, top1=top1))

def validate(val_loader, _model, criterion, epoch):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    _model.eval()

    end = time.time()
    for i, _ in enumerate(val_loader):
        input, target = _
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        output = _model(input)
        bs = target.size(0)
        loss = criterion(output, target)

        # measure accuracy and record loss
        accu = utils.accuracy(output.data, target)
        losses.update(loss.data, bs)
        top1.update(accu, bs)

        # # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accu {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Accu {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

if __name__ == "__main__":
    main()
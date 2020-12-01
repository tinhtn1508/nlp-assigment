import utils
import model
import torch
import torch.nn as nn
import time
import os
import data
import math
from loguru import logger

args = utils.get_train_parser()
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        logger.warning("You have a CUDA device, so you should probably run with --cuda")

device = torch.device('cuda' if args.cuda else 'cpu')


def train(_model, criterion, train_data, ntokens, learning_rate, epoch):
    _model.train()
    total_loss = .0
    start_time = time.time()
    hidden = _model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.sequence_length)):
        data, targets = utils.get_batch(train_data, i, min(args.sequence_length, len(train_data) - 1 - i))
        _model.zero_grad()
        hidden = utils.repackage_hidden(hidden)
        output, hidden = _model(data, hidden)

        loss = criterion(output, targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(_model.parameters(), 0.25)
        for p in _model.parameters():
            p.data.add_(p.grad, alpha=-learning_rate)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.sequence_length, learning_rate,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def validate(_model, criterion, valid_data, eval_batch_size):
    _model.eval()
    total_loss = .0
    hidden = _model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, valid_data.size(0) - 1, args.sequence_length):
            data, targets = get_batch(valid_data, i, min(args.sequence_length, len(valid_data) - 1 - i))
            output, hidden = _model(data, hidden)
            hidden = utils.repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(valid_data) - 1)

def test(criterion, test_data, eval_batch_size):
    with open(args.save, 'rb') as f:
        _model = torch.load(f)
        _model.rnn.flatten_parameters()
        test_loss = evaluate(_model, criterion, test_data, eval_batch_size)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)

def main():
    corpus = data.Corpus(args.data)
    eval_batch_size = 10
    train_data = utils.batchify(corpus.train, args.batch_size, device)
    val_data = utils.batchify(corpus.valid, eval_batch_size, device)
    test_data = utils.batchify(corpus.test, eval_batch_size, device)
    ntokens = len(corpus.dictionary)

    _model = model.RNNModel(args.model, ntokens, args.embsize, args.nhidden, args.nlayers, args.dropout).to(device)
    _criterion = nn.NLLLoss()
    _lr = args.lr
    best_val_loss = None

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(_model, _criterion, train_data, ntokens, _lr, epoch)
            val_loss = evaluate(_model, _criterion, val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                    best_val_loss = val_loss
            else:
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    test(_criterion, test_data, eval_batch_size)


if __name__ == "__main__":
    main()
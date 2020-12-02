import argparse
import torch
import data
import utils

args = utils.get_check_grammar_parser()
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        logger.warning("You have a CUDA device, so you should probably run with --cuda")

device = torch.device('cuda' if args.cuda else 'cpu')

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.CorpusCharacter(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)

chars = [c for c in args.prefix ]
encode = [corpus.dictionary.word2idx[i] for i in args.prefix]

head = [encode[0]]
tail = encode[1:len(encode)]
with torch.no_grad():
    while len(tail) != 0:
        input = torch.tensor([head], dtype=torch.long).t().to(device)
        next_idx = tail.pop(0)
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().div(1).exp().cpu()

        prob_next_char =  word_weights[next_idx].item() if word_weights.size(0) == 171 else word_weights[len(input)-1][next_idx].item()
        candidate = []
        if prob_next_char < 0.002:
            if word_weights.size(0) == 171:
                top5_value, top5_idx = torch.topk(word_weights, k=5)
            else:
                top5_value, top5_idx = torch.topk(word_weights[len(input)-1], k=5)

            candidate = ", ".join([corpus.dictionary.idx2word[c] for c in top5_idx])
            print('{} --> {} -- probablity: {:.2f}% -- candidate: {}'.format(chars.pop(0), chars[0], 100 * prob_next_char, candidate))
        else:
            print('{} --> {} -- probablity: {:.2f}%'.format(chars.pop(0), chars[0], 100 * prob_next_char))

        head.append(next_idx)
        input = torch.tensor([head], dtype=torch.long).to(device)

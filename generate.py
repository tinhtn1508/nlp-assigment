import argparse
import torch
import data
import utils

args = utils.get_generate_parser()
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        logger.warning("You have a CUDA device, so you should probably run with --cuda")

device = torch.device('cuda' if args.cuda else 'cpu')

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)

if args.prefix == '':
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
else:
    encode = [corpus.dictionary.word2idx[i] for i in args.prefix.split()]
    input = torch.tensor([encode], dtype=torch.long).to(device)
    input = torch.transpose(input, 0, 1)

input_size = input.size(0)
final_output = args.prefix + " "
nsentence = 0

with torch.no_grad():
    while True:
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().div(1.0).exp().cpu()
        top10_value, top10_idx = torch.topk(word_weights[input_size-1], k=100)
        word_idx = top10_idx[torch.multinomial(top10_value, 1)[0]]

        for i in range(0, input_size):
            if i == input_size - 1:
                input[i] = torch.tensor([word_idx], dtype=torch.long, device=device)
            else:
                input[i] = input[i+1]
        word = corpus.dictionary.idx2word[word_idx]
        if word == '<eos>':
            final_output += ".\n"
            nsentence += 1
        else:
            final_output += word + " "

        if nsentence == args.nsentence:
            print(final_output)
            break

import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.nn import functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, vocab_size: int, tagset_size: int):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)[:, -1, :]
        # tag_score = F.softmax(tag_space, dim=1)
        return tag_space

class RNNModel(nn.Module):
    def __init__(self, rnn_type, ntoken, nembedding, nhidden, nlayers, dropout=0.5):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, nembedding)
        self.rnn = getattr(nn, rnn_type)(nembedding, nhidden, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhidden, ntoken)
        self.rnn_type = rnn_type
        self.nhidden = nhidden
        self.nlayers = nlayers

        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhidden),
                    weight.new_zeros(self.nlayers, bsz, self.nhidden))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhidden)
    
    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

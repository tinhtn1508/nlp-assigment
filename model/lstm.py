import torch
import torch.nn as nn
import torch.tensor as tensor
from torch.nn import functional as F

class SimpleLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence)), 1, -1)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_score = F.log_softmax(tag_space, dim = 1)
        return tag_score
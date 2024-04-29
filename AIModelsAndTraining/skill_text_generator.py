import time
import operator
import torch.nn as nn
import torch

class GRUSkill(nn.Module):
    def __init__(self, word_dim,num_layers, hidden_dim=256, embedding_dim = 300):
        super(GRUSkill, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Define your layers here
        self.embedding = nn.Embedding(word_dim, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_dim, word_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        h_t, _ = self.gru(embedded)
        h_t = self.relu(h_t)
        output = self.out(h_t)
        #output = torch.nn.functional.softmax(output,dim=1)
        return output

    def predict(self,o):
        return torch.argmax(o,dim=1)

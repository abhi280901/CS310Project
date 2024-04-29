#!/usr/bin/env python3
import time
import operator
import torch.nn as nn
import torch
import numpy as np
import random
import math



class GRUPy(nn.Module):
    def __init__(self, word_dim,num_layers, hidden_dim=256, embedding_dim=200):
        super(GRUPy, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(num_embeddings=word_dim,embedding_dim = embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.rel2 = nn.LeakyReLU(0.2)
        self.hl2 = nn.Linear(hidden_dim, 512)
        self.rel3 = nn.LeakyReLU(0.2)
        self.output = nn.Linear(512, word_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        h_t, _ = self.gru(embedded)
        h_t = self.rel2(h_t)
        h_t = self.hl2(h_t)
        h_t = self.rel3(h_t)
        output = self.output(h_t)
        return output

    def predict(self,o):
        L = np.argsort(-o.detach(), axis=1)
        return [L[:,0],L[:,1],L[:,2]]
    
    def generate_sent(self,word,max_len):
        sent = np.zeros(max_len)
        sent[0] = word
        if sent[0] == 6:
            num = 2
        else: 
            num = 0
        for i in range(max_len-1):
            if i == 2:
                num = 0
            pr = self.forward(word)
            sent[i+1] = self.predict(pr)[num]
            word = torch.IntTensor([sent[i+1]])
        return sent
    

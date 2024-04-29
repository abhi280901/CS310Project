#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
from utils import *
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, word_dim, embedding_matrix,hidden_dim=256, embedding_dim=50):
        super(Classifier, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Define your layers here
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=True)
        self.hid = nn.Linear(embedding_dim,hidden_dim)
        self.relu = nn.ReLU()
        self.hid2 = nn.Linear(hidden_dim,128)
        self.linear = nn.Linear(128, 5)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling across the sequence

    def forward(self, x):
        embedded = self.embedding(x)
        h1_t = self.hid(embedded)
        h1_t = self.relu(h1_t)
        h2_t = self.hid2(h1_t)
        h2_t = h2_t.permute(0, 2, 1)  # Change the dimension for the pooling layer
        pooled = self.avg_pool(h2_t).squeeze(2)
        output = self.linear(pooled)
        return output
    
    def give_scale(self,x):
        x = torch.reshape(x,(1,x.shape[0]))
        output = self.forward(x)
        ar = np.argmax(output.detach())
        if ar == 0:
            scale = -2
        elif ar == 1:
            scale = -1
        elif ar == 2:
            scale = 0
        elif ar == 3:
            scale = 1
        else:
            scale = 2
        return scale

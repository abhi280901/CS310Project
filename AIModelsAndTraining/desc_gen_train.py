import torch
import numpy as np
import nltk
import pandas as pd
from utils import *
from sklearn.model_selection import KFold
import itertools
import torch.nn as nn
import matplotlib.pyplot as plt

VOCABULARY_SIZE = 991
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
DATAFILE = 'data/pokes_newest.xlsx'

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading Excel file...")
i = 0
df = pd.read_excel(DATAFILE)
df = df.reset_index(drop=True)
desc = df[["skill_text"]]


# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, y.iloc(0).lower(), SENTENCE_END_TOKEN) for x,y in desc.iterrows()]

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
print("Parsed %d sentences." % (len(tokenized_sentences)))

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))


# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(VOCABULARY_SIZE)
index_to_word = [x[0] for x in vocab]
index_to_word.append(UNKNOWN_TOKEN)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

print("Using vocabulary size %d." % VOCABULARY_SIZE)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

print("\nExample sentence: '%s'" % sentences[0])
print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Create the training data
X_train = [torch.IntTensor([word_to_index[w] for w in sent[:-1]]) for sent in tokenized_sentences]
y_train = [torch.IntTensor([word_to_index[w] for w in sent[1:]]) for sent in tokenized_sentences]
# Print an training data example
x_example, y_example = X_train[100], y_train[100]
print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

# Embedding code. Comment out to use embedding. Additionally set emb parameter within the Generator class below to embedding matrix.
'''
GLOVE_FILE = "data/glove.6B.50d.txt"
EMBEDDING_DIM = 50

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def create_embedding_matrix(word_to_index, embeddings, embedding_dim):
    vocab_size = len(word_to_index) + 1  # Add 1 for the unknown token
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, index in word_to_index.items():
        if word in embeddings:
            embedding_matrix[index] = embeddings[word]

    return embedding_matrix


glove_embeddings = load_glove_embeddings(GLOVE_FILE)
embedding_matrix = create_embedding_matrix(word_to_index, glove_embeddings, EMBEDDING_DIM)
embedding_matrix = torch.Tensor(embedding_matrix)
print(embedding_matrix.shape)
'''

import time
import operator
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, word_dim,num_layers, hidden_dim=512, embedding_dim=200,emb = None):
        super(Generator, self).__init__()
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Pick between pre-trained or self learned embedding layers.
        #self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=True)
        self.embedding = nn.Embedding(num_embeddings=word_dim,embedding_dim = embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.rel1 = nn.LeakyReLU()
        self.output = nn.Linear(hidden_dim, word_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        h_t, _ = self.gru(embedded)
        h_t = self.rel1(h_t)
        output = self.output(h_t)
        return output

    def predict(self,o):
        return torch.argmax(o,dim=1)

    def unfreeze_embedding(self):
        # Set the freeze parameter to False to unfreeze the embedding matrix
        self.embedding.weight.requires_grad = True

model = Generator(991,2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# COMPARISON TRAINING
# Training loop for k=4 folds cross validation, for 10 epochs per fold. The model is reinitialized after every fold.
# The training performance of the model is used for comparison amongst its variations and other models.
num_epochs = 10
arr = np.ones((40,5))
arr[0:40,0] = np.arange(1,41)
k = 4
kf = KFold(n_splits=k, shuffle=True)

for fold, (train_indices, val_indices) in enumerate(kf.split(X_train)):
    model = Generator(991,1)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    X_train_subset = [X_train[ind] for ind in train_indices]
    X_test_subset = [X_train[ind] for ind in val_indices]
    y_train_subset = [y_train[ind] for ind in train_indices]
    y_test_subset = [y_train[ind] for ind in val_indices]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_preds = 0
        
        for i in range(len(X_train_subset)):
            # Forward pass
            inputs = X_train_subset[i]
            targets = y_train_subset[i]
            output = model.forward(inputs)
            predictions = model.predict(output)
            targets = targets.type(torch.LongTensor)
            # Calculate the loss
            loss = criterion(output, targets)
            total_preds += len(X_train_subset[i])
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            total_correct += (predictions == targets).sum().item()
            if epoch > 0:
                if (total_loss / (len(X_train))) > los:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']*0.5
                        
                        
        accuracy = total_correct / total_preds
        los = total_loss / (len(X_train_subset))
        # Evaluation phase on the test set
        model.eval()
        test_total_loss = 0
        test_total_correct = 0
        test_total_preds = 0
        with torch.no_grad():  # Disable gradient computation for evaluation
            for i in range(len(X_test_subset)):
                inputs = X_test_subset[i]
                targets = y_test_subset[i]
                output = model.forward(inputs)
                predictions = model.predict(output)
                targets = targets.type(torch.LongTensor)
                loss = criterion(output, targets)
                test_total_preds += len(X_test_subset[i])
                test_total_loss += loss.item()
                test_total_correct += (predictions == targets).sum().item()
    
        # Compute test accuracy
        test_accuracy = test_total_correct / test_total_preds
        test_loss = test_total_loss / len(X_test_subset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {los}, Accuracy: {accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        
        arr[epoch + (5*fold),1] = total_loss / (len(X_train_subset))
        arr[epoch + (5*fold),2] = accuracy
        arr[epoch + (5*fold),3] = test_loss
        arr[epoch + (5*fold),4] = test_accuracy

# PURE PERFORMANCE TRAINING
# Training loop showcasing the true potential of the model. Model is not reinitialized and trained over 20 epochs.
# The training performance of the model is used for mapping the model's pure performnce over 20 epochs. Uncomment to use.
'''
num_epochs = 5
arr = np.ones((20,5))
arr[0:20,0] = np.arange(1,21)
k = 4
kf = KFold(n_splits=k, shuffle=True)

for fold, (train_indices, val_indices) in enumerate(kf.split(X_train)):
    X_train_subset = [X_train[ind] for ind in train_indices]
    X_test_subset = [X_train[ind] for ind in val_indices]
    y_train_subset = [y_train[ind] for ind in train_indices]
    y_test_subset = [y_train[ind] for ind in val_indices]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_preds = 0
        
        for i in range(len(X_train_subset)):
            # Forward pass
            inputs = X_train_subset[i]
            targets = y_train_subset[i]
            output = model.forward(inputs)
            predictions = model.predict(output)
            targets = targets.type(torch.LongTensor)
            # Calculate the loss
            loss = criterion(output, targets)
            total_preds += len(X_train_subset[i])
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            total_correct += (predictions == targets).sum().item()
            if epoch > 0:
                if (total_loss / (len(X_train))) > los:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']*0.5
                        
                        
        accuracy = total_correct / total_preds
        los = total_loss / (len(X_train_subset))
        # Evaluation phase on the test set
        model.eval()
        test_total_loss = 0
        test_total_correct = 0
        test_total_preds = 0
        with torch.no_grad():  # Disable gradient computation for evaluation
            for i in range(len(X_test_subset)):
                inputs = X_test_subset[i]
                targets = y_test_subset[i]
                output = model.forward(inputs)
                predictions = model.predict(output)
                targets = targets.type(torch.LongTensor)
                loss = criterion(output, targets)
                test_total_preds += len(X_test_subset[i])
                test_total_loss += loss.item()
                test_total_correct += (predictions == targets).sum().item()
    
        # Compute test accuracy
        test_accuracy = test_total_correct / test_total_preds
        test_loss = test_total_loss / len(X_test_subset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {los}, Accuracy: {accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        
        arr[epoch + (5*fold),1] = total_loss / (len(X_train_subset))
        arr[epoch + (5*fold),2] = accuracy
        arr[epoch + (5*fold),3] = test_loss
        arr[epoch + (5*fold),4] = test_accuracy
'''
# Plot performance, typically used after pure performance training.
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Loss and Accuracy plots')
ax1.plot(arr[0:20,0],arr[0:20,1],label="Training Loss")
ax1.plot(arr[0:20,0],arr[0:20,3],label="Evaluation Loss")
ax1.set(xlabel='Epochs', ylabel='Loss')
ax1.label_outer()
ax1.legend()
ax2.plot(arr[0:20,0],arr[0:20,2],label="Training Accuracy")
ax2.plot(arr[0:20,0],arr[0:20,4],label="Evaluation Accuracy")
ax2.set(xlabel='Epochs', ylabel='Accuracy')
ax2.legend()
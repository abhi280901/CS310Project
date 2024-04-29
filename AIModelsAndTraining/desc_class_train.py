from utils import *
import itertools
import nltk
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import desc_classifier as dc
import torch.nn as nn

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
power_scale = df[["power_scale"]]


# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, y.iloc[0].lower(), SENTENCE_END_TOKEN) for x,y in desc.iterrows()]

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
i = 0
y_train = [0]*len(X_train)
for x,y in power_scale.iterrows():
    if y["power_scale"] == -2:
        y_train[i] =  torch.IntTensor([1,0,0,0,0])
    elif y["power_scale"] == -1:
        y_train[i] =  torch.IntTensor([0,1,0,0,0])
    elif y["power_scale"] == 0:
        y_train[i] =  torch.IntTensor([0,0,1,0,0])
    elif y["power_scale"] == 1:
        y_train[i] =  torch.IntTensor([0,0,0,1,0])
    elif y["power_scale"] == 2:
        y_train[i] =  torch.IntTensor([0,0,0,0,1])
    i=i+1
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Print an training data example
x_example, y_example = X_train[0], y_train[0]
#print(X_train)
print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
print("\ny:\n", y_example)
print("\n")



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

model = dc.Classifier(991,embedding_matrix)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10
arr = np.ones((num_epochs,5))
arr[0:num_epochs,0] = np.arange(1,num_epochs+1)
for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_preds = 0
        
        for i in range(len(X_train)):
            # Forward pass
            inputs = X_train[i]
            targets = y_train[i]
            inputs = torch.reshape(inputs,(1,inputs.shape[0]))
            output = model.forward(inputs)
            targets = targets.type(torch.FloatTensor)
            targets = torch.reshape(targets,(1,5))
            # Calculate the loss
            loss = criterion(output, targets)
            total_preds += 1
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            ar = np.argmax(output.detach())
            arg = np.argmax(targets.detach())
            if (ar == arg):
                total_correct += 1
            if epoch > 0:
                if (total_loss / (len(X_train))) > los:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']*0.5
                        
                        
        accuracy = total_correct / total_preds
        los = total_loss / (len(X_train))
        # Evaluation phase on the test set
        model.eval()
        test_total_loss = 0
        test_total_correct = 0
        test_total_preds = 0
        with torch.no_grad():  # Disable gradient computation for evaluation
            for i in range(len(X_test)):
                inputs = X_test[i]
                targets = y_test[i]
                inputs = torch.reshape(inputs,(1,inputs.shape[0]))
                output = model.forward(inputs)
                targets = targets.type(torch.FloatTensor)
                targets = torch.reshape(targets,(1,5))
                # Calculate the loss
                loss = criterion(output, targets)
                test_total_preds += 1
                # Backward pass
                test_total_loss += loss.item()
                ar = np.argmax(output.detach())
                arg = np.argmax(targets.detach())
                if (ar == arg):
                    test_total_correct += 1
    
        # Compute test accuracy
        test_accuracy = test_total_correct / test_total_preds
        test_loss = test_total_loss / len(X_test)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {los}, Accuracy: {accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')
        
        arr[epoch,1] = total_loss / (len(X_train))
        arr[epoch,2] = accuracy
        arr[epoch,3] = test_loss
        arr[epoch,4] = test_accuracy

        

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Loss and Accuracy plots')
ax1.plot(arr[0:num_epochs,0],arr[0:num_epochs,1],label="Training Loss")
ax1.plot(arr[0:num_epochs,0],arr[0:num_epochs,3],label="Evaluation Loss")
ax1.set(xlabel='Epochs', ylabel='Loss')
ax1.label_outer()
ax1.legend()
ax2.plot(arr[0:num_epochs,0],arr[0:num_epochs,2],label="Training Accuracy")
ax2.plot(arr[0:num_epochs,0],arr[0:num_epochs,4],label="Evaluation Accuracy")
ax2.set(xlabel='Epochs', ylabel='Accuracy')
ax2.legend()
plt.show()
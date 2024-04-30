import torch
import torch.nn as nn
import numpy as np
import nltk
import pandas as pd
import itertools
from utils import *
from sklearn.model_selection import KFold
import skill_text_generator as stg
import matplotlib.pyplot as plt

VOCABULARY_SIZE = 1400
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START_TOKEN = "WORD_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
DATAFILE = 'data/skill_name.xlsx'

# Read the data and append SENTENCE_START and SENTENCE_END tokens
print("Reading Excel file...")
i = 0
df = pd.read_excel(DATAFILE)
df = df.dropna()
df = df.reset_index(drop=True)


# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, y.iloc[0].lower(), SENTENCE_END_TOKEN) for x,y in df.iterrows()]

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
x_example, y_example = X_train[2], y_train[2]
#print(X_train)
print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
print("\ny:\n%s\n%s\n" % (" ".join([index_to_word[x] for x in y_example]), y_example))

model = stg.GRUSkill(2031,1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# PERFORMANCE TRAINING
# Training loop showcasing the true potential of the model. Model is not reinitialized and trained over 20 epochs.
# The training performance of the model is used for mapping the model's pure performance over 20 epochs.

# COMPARISON TRAINING
# Training loop for k=4 folds cross validation, for 10 epochs per fold. The model is reinitialized after every fold.
# The training performance of the model is used for comparison amongst its variations and other models. 

# Uncomment lines 87-89 to change from performance to comparison.

# change num of epochs and folds appropriately
#model.load_state_dict(torch.load('Parameters/names_card_gru_relu.txt'))
num_epochs = 10
k = 4
runs = num_epochs*k
arr = np.ones((runs,5))
arr[0:runs,0] = np.arange(1,runs + 1)

kf = KFold(n_splits=k, shuffle=True)

for fold, (train_indices, val_indices) in enumerate(kf.split(X_train)):
    #model = stg.GRUSkill(2031,1)
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
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
        
        arr[epoch + (num_epochs*fold),1] = total_loss / (len(X_train_subset))
        arr[epoch + (num_epochs*fold),2] = accuracy
        arr[epoch + (num_epochs*fold),3] = test_loss
        arr[epoch + (num_epochs*fold),4] = test_accuracy
#torch.save(model.state_dict(), 'Parameters/names_card_gru_relu.txt')

# Plot performance, typically used after pure performance training.
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Loss and Accuracy plots')
ax1.plot(arr[0:runs,0],arr[0:runs,1],label="Training Loss")
ax1.plot(arr[0:runs,0],arr[0:runs,3],label="Evaluation Loss")
ax1.set(xlabel='Epochs', ylabel='Loss')
ax1.label_outer()
ax1.legend()
ax2.plot(arr[0:runs,0],arr[0:runs,2],label="Training Accuracy")
ax2.plot(arr[0:runs,0],arr[0:runs,4],label="Evaluation Accuracy")
ax2.set(xlabel='Epochs', ylabel='Accuracy')
ax2.legend()
plt.show()
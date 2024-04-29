#!/usr/bin/env python3
import desc_classifier as dc
import attribute_generator as ag
import desc_generator as dg
import torch
import numpy as np
import nltk
import pandas as pd
import itertools
import torch.nn as nn
import csv
import skill_text_generator as stg
import random

VOCABULARY_SIZE = 991
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START_TOKEN = "SENTENCE_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
DATAFILE = 'data/pokes_newest.xlsx'
GLOVE_FILE = "data/glove.6B.50d.txt"
EMBEDDING_DIM = 50

def create_training_data():

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    #print("Reading Excel file...")
    i = 0
    df = pd.read_excel(DATAFILE)
    df = df.reset_index(drop=True)
    desc = df[["skill_text"]]
    power_scale = df[["power_scale"]]

    #print(desc)
    #print(power_scale)


    # Append SENTENCE_START and SENTENCE_END
    sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, y.iloc[0].lower(), SENTENCE_END_TOKEN) for x,y in desc.iterrows()]

    #print(sentences)

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    #print("Parsed %d sentences." % (len(tokenized_sentences)))

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    #print("Found %d unique words tokens." % len(word_freq.items()))


    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(VOCABULARY_SIZE)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(UNKNOWN_TOKEN)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    #print("Using vocabulary size %d." % VOCABULARY_SIZE)
    #print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

    #print("\nExample sentence: '%s'" % sentences[0])
    #print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
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
    # Print an training data example
    x_example, y_example = X_train[0], y_train[0]
    return X_train,y_train,word_to_index,index_to_word

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




X_train,y_train,word_to_index,index_to_word = create_training_data()

glove_embeddings = load_glove_embeddings(GLOVE_FILE)
embedding_matrix = create_embedding_matrix(word_to_index, glove_embeddings, EMBEDDING_DIM)
embedding_matrix = torch.Tensor(embedding_matrix)

### DESCRIPTION GENERATOR ###
num = random.randint(0,1)
if num == 0:
    word = "heal"
elif num ==1:
    word = "attack"
generator = dg.GRUPy(991,1)

generator.eval()
#initialising model weights to learned parameters
generator.load_state_dict(torch.load('Parameters/desc_gen_leakyrelu_noemb.txt'))
sent = generator.generate_sent(torch.IntTensor([int (word_to_index[word])]),9)
sent = torch.IntTensor([int (x) for x in sent])

### DESCRIPTION CLASSIFIER ###



model = dc.Classifier(991,embedding_matrix)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.eval()
model.load_state_dict(torch.load('Parameters/1st_try_2hl_relu.txt'))
print([index_to_word[x] for x in sent])
pow_scale = model.give_scale(sent)

### ATTRIBUTE GENERATOR ###

ps0,ps1,ps2,psneg1,psneg2 = ag.read(DATAFILE)

sample = [[-1,0,-1]]
while sample[0][0] <= 0 or sample[0][1] < 1 or sample[0][2] <= 0:
    if pow_scale == 0:
        sample = ag.sample(ps0)
    elif pow_scale == 1:
        sample = ag.sample(ps1)
    elif pow_scale == 2:
        sample = ag.sample(ps2)
    elif pow_scale == -1:
        sample = ag.sample(psneg1)
    elif pow_scale == -2:
        sample = ag.sample(psneg2)

skill_gen = stg.GRUSkill(2030,1)
skill_gen.eval()
skill_gen.load_state_dict(torch.load('Parameters/names_card_gru_relu.txt'))
num = random.randint(0,1399)

VOCABULARY_SIZE = 1400
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
SENTENCE_START_TOKEN = "WORD_START"
SENTENCE_END_TOKEN = "SENTENCE_END"
DATAFILE = 'data/skill_name.xlsx'

# Read the data and append SENTENCE_START and SENTENCE_END tokens
#print("Reading Excel file...")
i = 0
df = pd.read_excel(DATAFILE)
df = df.dropna()
df = df.reset_index(drop=True)


# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (SENTENCE_START_TOKEN, y.iloc[0].lower(), SENTENCE_END_TOKEN) for x,y in df.iterrows()]

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
#print("Parsed %d sentences." % (len(tokenized_sentences)))

# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
#print("Found %d unique words tokens." % len(word_freq.items()))


# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(VOCABULARY_SIZE)
index_to_word = [x[0] for x in vocab]
index_to_word.append(UNKNOWN_TOKEN)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

#print("Using vocabulary size %d." % VOCABULARY_SIZE)
#print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else UNKNOWN_TOKEN for w in sent]

#print("\nExample sentence: '%s'" % sentences[0])
#print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

inp = torch.IntTensor([num])
ma = True
while ma:
    num = skill_gen.forward(inp)
    softmaxed = torch.nn.functional.softmax(num,dim=1)
    pr = skill_gen.predict(softmaxed)
    for x in pr:
        if x == word_to_index["UNKNOWN_TOKEN"] or x == word_to_index["WORD_START"] or x== word_to_index["SENTENCE_END"] or x==word_to_index["\\n"] or x == word_to_index["power\\n"]:
            ma = True
            num = random.randint(0,1399)
            inp = torch.IntTensor([num])
            break
        else:
            ma = False

print(pow_scale)
print(sample)
print([index_to_word[inp]])
print([index_to_word[x] for x in pr])

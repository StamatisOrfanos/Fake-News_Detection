import gensim, time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd 
import gensim
import string, nltk
from nltk import word_tokenize
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten, GRU, LSTM
from tensorflow.keras.initializers import Constant
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

EMBEDDING_DIM = 128


# Tokenize the data in order to be used by both the Word2Vec model and the Tokenizer()
def getreviewLine(data):
    reviewLines = list()
    tokens = word_tokenize(data)
    reviewLines.append(tokens)
    return reviewLines

# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    print(f"The weight matrix initialy is {weight_matrix}")

    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


data = "in fake news detection we detect the fake news"
reviewLines = getreviewLine(data)


# Initialize the Word2Vec model and save the model to use it in the first layer of the NN
word2vecSteps = gensim.models.Word2Vec(sentences=reviewLines, size=EMBEDDING_DIM, window=5, min_count=1)
filename = 'word2vecSteps.txt'
word2vecSteps.wv.save_word2vec_format(filename, binary=False)


# Now we convert the word embedding to tokenized vector
tokenizer = Tokenizer(num_words=60000)
tokenizer.fit_on_texts(reviewLines)
sequences = tokenizer.texts_to_sequences(reviewLines)
wordIndex = tokenizer.word_index


reviewData = pad_sequences(sequences, maxlen=20, padding='post')
vocabSize = len(tokenizer.word_index) + 1        # Plus one for all the unknown words

#Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer
embedding_vectors = get_weight_matrix(word2vecSteps, wordIndex)


# Print the results
print('\n\n')
print(f"The result of the getReviewLine is: {reviewLines}")
print('\n\n')
print(f"The result of the Tokenizer is this sequence: {sequences}")
print('\n\n')
print(f"The word index is the following: {wordIndex}")
print('\n\n')
print(f"The vocabulary size we are going to use is {vocabSize}")
print('\n\n')
print(f"The data that we insert to the model for the training are: {reviewData}")
print('\n\n')
print(f"The weights we use in teh input layer are {embedding_vectors}")
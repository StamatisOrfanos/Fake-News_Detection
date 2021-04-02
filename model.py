import gensim, time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import gensim, string, nltk
from nltk import word_tokenize
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Global Variables
EMBEDDING_DIM = 128
max_length = 500

# Tokenize the data in order to be used by both the Word2Vec model and the Tokenizer()
def getreviewLine(data):
    reviewLines = list()
    lines = data['text'].to_list()

    for line in lines:
        tokens = word_tokenize(line)
        reviewLines.append(tokens)
    return reviewLines


# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab):
    vocab_size = len(vocab) + 1
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix



# Import the data
data = pd.read_csv('finalData/trainSet.csv')
data.dropna(subset=['text'], inplace=True)
reviewLines = getreviewLine(data)
label = data["label"].values


start = time.time()

# Initialize the Word2Vec model and save the model to use it in the first layer of the NN
word2vec = gensim.models.Word2Vec(sentences=reviewLines, size=EMBEDDING_DIM, window=5, min_count=1)
filename = 'word2vec.txt'
word2vec.wv.save_word2vec_format(filename, binary=False)



# Now we convert the word embedding to tokenized vector
tokenizer = Tokenizer(num_words=90000)
tokenizer.fit_on_texts(reviewLines)
sequences = tokenizer.texts_to_sequences(reviewLines)
wordIndex = tokenizer.word_index

reviewData = pad_sequences(sequences, maxlen=max_length, padding='post')
vocabSize = len(tokenizer.word_index) + 1


#Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer
embedding_vectors = get_weight_matrix(word2vec, wordIndex)


# Define the Sequential model
model = Sequential()

model.add(Embedding(vocabSize, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=max_length, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Train the model
trainDataF, valiData, trainLabel, valiLabel = train_test_split(reviewData, label)
history = model.fit(trainDataF, trainLabel, batch_size=128, epochs=5, validation_data=(valiData, valiLabel), verbose=1)

end = time.time()

# Write the time needed to both make and train the model
f = open('Time/time_needed.txt', 'a')
f.write(f"\n\nThe time to make the Word2Vec embeddig and NN model is: {end-start}")
f.close()

# Save the model for later use in the application 
model.save("model.h5")
print("Saved model to disk")
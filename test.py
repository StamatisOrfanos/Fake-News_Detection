import gensim, time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import gensim, string, nltk
from nltk import word_tokenize
import tensorflow as tf
from numpy import loadtxt
from tensorflow.keras.models import load_model
import tensorflow.keras as tfk
from tensorflow.keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# Global Variables 
max_length = 500


# Tokenize the data in order to be used by both the Word2Vec model and the Tokenizer()
def getreviewLine(data):
    reviewLines = list()
    lines = data['text'].to_list()

    for line in lines:
        tokens = word_tokenize(line)
        reviewLines.append(tokens)
    return reviewLines


# Import the data
data = pd.read_csv('finalData/testSet.csv')
data.dropna(subset=['text'], inplace=True)
testLines = getreviewLine(data)
labels = data["label"]


# Now we convert the word embedding to tokenized vector as an input for the model
tokenizer = Tokenizer(num_words=90000)
tokenizer.fit_on_texts(testLines)
sequences = tokenizer.texts_to_sequences(testLines)
wordIndex = tokenizer.word_index
reviewData = pad_sequences(sequences, maxlen=max_length, padding='post')



# Load the model
model = load_model("model.h5", compile=True)

start = time.time()

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(reviewData[:1000], labels[:1000], batch_size=128)
print("The first value is the test loos and the second the test accuracy:", results)

end = time.time()

# Write the time needed to predict for 1000 values
f = open('Time/time_needed.txt', 'a')
f.write(f"\n\nThe time to make 1000 predictions is: {end-start}")
f.close()
# print(model.predict(reviewData[[4000]]))
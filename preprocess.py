import pandas as pd
import numpy as np
import string, re, os, time, nltk
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
import pandas as pd 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


# Some characters to remove from the text that are not ascii and some variables for clean results
table = str.maketrans('', '', string.punctuation)
stop_words = stopwords.words('english')

# Standarization Function for the data
def standarizeData(data):
    newText = []

    data.dropna(subset=['text'], inplace=True)                       # Drop rows that are null
    data.drop(columns=['id', 'title', 'author'], inplace=True)       # Drop all the non usefull cols

    data['text'] = data['text'].str.lower()                          # Remove capital letters

    for i, row in data.iterrows():
        text = row['text']
        text = str(BeautifulSoup(text, features='html.parser'))      # Remove hmtl, xml tags from string
        tokens = word_tokenize(text)                                 # Tokenize data
        tokens = [word.translate(table) for word in tokens]          # Remove punctuation
        tokens = [word for word in tokens if word.isalpha()]         # Remove non-alphabetic
        tokens = [word for word in tokens if len(word) > 1]          # Remove tokens with lenght=1
        tokens_without_sw = [word for word in tokens if not word in stop_words] # Remove stopwords
        sentence = TreebankWordDetokenizer().detokenize(tokens_without_sw)

        
        newText.append([sentence, row['label']])

    data = pd.DataFrame(newText, columns=['text', 'label'])

    return data


# Create the train and test sets
def createTesting(inputData, percentage):
    train, test = train_test_split(inputData, test_size=percentage, shuffle=False)
    return train, test

# Import the data
inputData = pd.read_csv('data.csv')
print(len(inputData))

# Get the time to preprocess the data
start = time.time()
data = standarizeData(inputData)
end = time.time()

data.dropna(subset=['text'], inplace=True)
train, test = createTesting(data, 0.2)
train.to_csv('finalData/trainSet.csv')
test.to_csv('finalData/testSet.csv')

f = open('Time/time_needed.txt', 'a')
f.write(f"The time to process the data is: {end-start}")
f.close()

print(len(train))
print(len(test))
import io
import itertools
import numpy as np
import os
import re
import string
import tensorflow as tf
import tqdm

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Dot, Embedding, Flatten, GlobalAveragePooling1D, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# With negative sampling we are going to get one context_word withing the 
# neighborhood of the target_word and then we are going to choose random
# words from the vocabulary and we use a seed not to be completly random
SEED = 42 
AUTOTUNE = tf.data.experimental.AUTOTUNE




# Tokenize the sentence below
sentence = "The wide road shimmered in the hot sun"
print(sentence)
print("\n")
tokens = list(sentence.lower().split())


# Build the vocabulary and give each word and index
vocab, index = {}, 1 # start indexing from 1
vocab['<pad>'] = 0 # add a padding token 

for token in tokens:
	if token not in vocab:
		vocab[token] = index
		index += 1

vocab_size = len(vocab)
print(vocab)
print("\n")


# We also buid an inverse vocabulary 
inverse_vocab = {index: token for token, index in vocab.items()}


# We vectorize the sentence, based on the vocabulary we have
example_sequence = [vocab[word] for word in tokens]
print(example_sequence)


# Generate only positive skip-grams from a series of tokens with window size-> window_size
window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(example_sequence, 
	vocabulary_size=vocab_size,
	window_size=window_size,
	negative_samples=0)





# The skipgrams function returns all positive skip-gram pairs by sliding over a given window span.
# To produce additional skip-gram pairs that would serve as negative samples for training,
# you need to sample random words from the vocabulary. Since the sentece is really small
# the random words are going to be in the window of the target_word
# Rule --> Vocabulary is BIG   --> neg_samples = 2-5
#      --> Vocabulary is SMALL --> neg_samples = 5-20


# Get target and context words for one positive skip-gram. We are getting always the first one,
# which is (wide, the)
target_word, context_word = positive_skip_grams[0]

# Set the number of negative samples per positive context.
num_ns = 2

context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))
negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
	true_classes=context_class, # class that should be sampled as 'positive'
	num_true=1,                 # each positive skip-gram has 1 positive context class
	num_sampled=num_ns,         # number of negative context words to sample
	unique=True,                # all the negative samples should be unique
	range_max=vocab_size,       # pick index of the samples from [0, vocab_size]
	seed=SEED,                  # seed for reproducibility
	name="negative_sampling"    # name of this operation
)
print([inverse_vocab[index.numpy()] for index in negative_sampling_candidates])





# Construct one training example

# For a given positive (target_word, context_word) skip-gram, you now also have num_ns
# negative sampled context words that do not appear in the window size neighborhood of target_word.
# Batch the 1 positive context_word and num_ns negative context words into one tensor.
# This produces a set of positive skip-grams (labelled as 1) and negative samples (labelled as 0)
# for each target word.


# Add a dimension so you can use concatenation (on the next step, then concat positive context word 
# with negative sampled words.
negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
context = tf.concat([context_class, negative_sampling_candidates], 0)


# Label first context word as 1 (positive) followed by num_ns 0s (negative).
label = tf.constant([1] + [0]*num_ns, dtype="int64") 

# Reshape target to shape (1,) and context and label to (num_ns+1,).
target = tf.squeeze(target_word)
context = tf.squeeze(context)
label =  tf.squeeze(label)


print(f"target_index    : {target}")
print(f"target_word     : {inverse_vocab[target_word]}")
print(f"context_indices : {context}")
print(f"context_words   : {[inverse_vocab[c.numpy()] for c in context]}")
print(f"label           : {label}")
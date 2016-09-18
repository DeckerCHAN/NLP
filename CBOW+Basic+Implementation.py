# coding: utf-8

# Logistic Regression with Bag-of-words features.
# -----------------------------------------------
# 
# This is a basic implementation of CBOW without considering scalability issue. A more scalable implementation can be found at <a \href="https://github.com/tensorflow/tensorflow/blob/r0.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py">word2vec basic implementation on Tensorflow</a>

# In[1]:

import tensorflow as tf

# In[2]:

import numpy as np

# In[3]:

from nltk import word_tokenize

# In[4]:

from random import shuffle

# In[5]:

train_ngrams = ['study in the united states',
                'live in new zealand',
                'study in canada UNK canadian',
                'study in the january NUM',
                'stay in the new england',
                'study in the december NUM',
                'study in the us PUN',
                'live in the us END_S',
                'study in the united kingdom'
                'live in the usa PUN'
                'study in clinical trials PUN',
                'work in the united states',
                'work in australia PUN',
                'its meeting on NUM january',
                'annual meeting on NUM december',
                'a meeting on january NUM',
                'ordinary meeting on NUM december',
                'regular meeting of february NUM']

# In[6]:

import collections
from collections import namedtuple

Ngram = namedtuple('Ngram', 'context c_word')


# In[7]:

def tokenize(ngram):
    return word_tokenize(ngram)


# Map each word to its ID and build the reverse map.

# In[8]:

def build_vocab(train_set):
    words = list()
    for ngram in train_set:
        tokens = tokenize(ngram)
        words.extend(tokens)
    count = collections.Counter(words).most_common()
    word_to_id = dict()
    word_to_id['NULL'] = 0
    for word, _ in count:
        word_to_id[word] = len(word_to_id)
    id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
    return word_to_id, id_to_word


# In[9]:

def map_token_seq_to_word_id_seq(token_seq, word_to_id):
    return [map_word_to_id(word_to_id, word) for word in token_seq]


def map_word_to_id(word_to_id, word):
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['NULL']


# Build a training dataset. Each instance(data point) consists of the ID of a word and the IDs of its context words.

# In[10]:

def build_dataset(train_set, word_to_id, window_size):
    dataset = list()
    for ngram in train_set:
        tokens = tokenize(ngram)
        word_id_seq = map_token_seq_to_word_id_seq(tokens, word_to_id)
        for i in range(len(word_id_seq)):
            word_context = [word_to_id['NULL']] * 2 * window_size
            for p_in_context in range(window_size):
                # position to the left of the current word in the given ngram
                p_left_ngram = i - window_size + p_in_context
                if p_left_ngram >= 0:
                    word_context[p_in_context] = word_id_seq[p_left_ngram]
                # position to the right of the current word in the given ngram
                p_right_ngram = i + p_in_context + 1
                if p_right_ngram < len(word_id_seq):
                    word_context[p_in_context + window_size] = word_id_seq[p_right_ngram]
            # word_context is the list of context word ids. c_word is the id of the current word.
            ngram_inst = Ngram(context=word_context, c_word=word_id_seq[i])
            dataset.append(ngram_inst)
    return dataset


# In[11]:

def print_dataset(dataset, id_to_word):
    for inst in dataset:
        print("%s : %s" % ([id_to_word[id] for id in inst.context], id_to_word[inst.c_word]))


# In[12]:




# Convert label y = word_id into its vector format with 1-of-K encoding.  

# In[13]:

def convert_to_label_vec(word_id, num_words):
    # initialise a zero vector of the length num_words
    label_vec = [0] * num_words
    label_vec[word_id] = 1
    return label_vec


# In[14]:

def train_eval(word_to_id, train_dataset, dev_dataset, num_epochs=10, learning_rate=0.1, embedding_dim=10):
    num_words = len(word_to_id)
    # Placeholders are inputs of the computation graph. 
    input_ngram = tf.placeholder(tf.int32, shape=[None])
    correct_label = tf.placeholder(tf.float32, shape=[num_words])
    # Word embeddings are the only parameters of the model
    embeddings = tf.Variable(tf.random_uniform([num_words, embedding_dim], -1.0, 1.0))
    # bias is not needed for embeddings
    # b = tf.Variable(tf.zeros([num_words]))

    with tf.Session() as sess:
        embed = tf.nn.embedding_lookup(embeddings, input_ngram)
        tmp_m = tf.reduce_sum(embed, 0)
        sum_rep = tf.reshape(tmp_m, [1, embedding_dim])
        # Formulate word embedding learning as a word prediction task. Note that, no negative sampling is applied here.
        y = tf.nn.softmax(tf.matmul(sum_rep, embeddings, transpose_b=True))
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_label * tf.log(y), reduction_indices=[1]))

        # evaluation code
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(correct_label, 0))
        accuracy = tf.cast(correct_prediction, tf.float32)

        sess.run(tf.initialize_all_variables())
        # Build SGD optimizer
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
        for epoch in range(num_epochs):
            shuffle(train_dataset)
            for ngram_inst in train_dataset:
                # Run one step of SGD to update word embeddings.
                train_step.run(feed_dict={input_ngram: ngram_inst.context,
                                          correct_label: convert_to_label_vec(ngram_inst.c_word, num_words)})
            # demonstrate the learning process of showing training accuracy.
            # If early stopping is desired, a validation set should be provided here instead of the train set. 
            # A simple heuristic rule for early stopping :  
            # Stop after accuracy on the validation set keep decreasing m epochs.   
            print('Epoch %d : %s .' % (
                epoch, compute_accuracy(num_words, accuracy, input_ngram, correct_label, dev_dataset)))
    return embeddings


# In[15]:

def compute_accuracy(num_words, accuracy, input_ngram, correct_label, eval_dataset):
    num_correct = 0
    for ngram_inst in eval_dataset:
        num_correct += accuracy.eval(feed_dict={input_ngram: ngram_inst.context,
                                                correct_label: convert_to_label_vec(ngram_inst.c_word, num_words)})
    print('#correct words is %s ' % num_correct)
    return num_correct / len(eval_dataset)


# In[16]:
word_to_id, id_to_word = build_vocab(train_ngrams)
train_set = build_dataset(train_ngrams, word_to_id, 2)
print_dataset(train_set, id_to_word)

learned_embeddings = train_eval(word_to_id, train_set, train_set)
print(learned_embeddings)

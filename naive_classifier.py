#!/usr/bin/env python
from mtg_data import load_card_data
from w2v_mtg import MTGTokenizer

train, test = load_card_data()

print len(train), len(test)
print train[0]


def prepare_onehot_vectors():
    from keras.preprocessing.text import Tokenizer
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder

    vocabulary_size = 2000
    #keep commas and colons
    train, test = load_card_data()
    corpus = [t[2] for t in train]
    #corpus = [line.strip().lower().replace("(", "").replace(")", "").replace(".", "").split() for line in open('corpus.txt', 'r')]
    tokenizer = MTGTokenizer(nb_words=vocabulary_size, filters=None, lower=True, split=" ")
    tokenizer.fit_on_texts(corpus)
    tokens = tokenizer.texts_to_sequences(corpus)

    print np.asarray(tokens)[0]
    OneHotEncoder(categorical_features="array of indices")

    X_train = []
    y_train = []
    """
        couples is a list of 2-elements lists of int: [word_index, other_word_index].
        labels is a list of 0 and 1, where 1 indicates that other_word_index was found in the same window as word_index, and 0 indicates that other_word_index was random.
    """

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

prepare_onehot_vectors()

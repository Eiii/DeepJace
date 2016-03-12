#!/usr/bin/env python
from mtg_data import load_card_data, load_set_data
from w2v_mtg import MTGTokenizer
import numpy as np
from keras.models import Graph, Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.core import Merge, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from scipy.sparse import csc_matrix
from sklearn.preprocessing import OneHotEncoder
import theano
import theano.tensor as T

VOCAB_SIZE = 2000
MAX_LEN = 40
DROPOUT = 0.7

def build_language_model():
  model = Sequential()
  model.add(Embedding(VOCAB_SIZE+1, 512, mask_zero=True, input_length=MAX_LEN)) #vocab, size
  model.add(LSTM(64))
  model.add(Dropout(DROPOUT))
  return model

def build_numeric_model(input_shape):
  model = Sequential()
  model.add(Dense(64, input_shape=input_shape, activation='relu'))
  model.add(Dropout(DROPOUT))
  return model

def build_full_model(input_shape, pretrain_language=None):
  if pretrain_language is None:
    language_model = build_language_model()
  else:
    language_model = pretrain_language
    language_model.layers.pop()
  numeric_model = build_numeric_model(input_shape)
  model = Sequential()
  model.add(Merge([language_model, numeric_model], mode='concat', concat_axis=-1))
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(DROPOUT))
  model.add(Dense(6, activation='relu'))
  return model

def build_pretrain_model():
  model = build_language_model()
  model.add(Dense(1, activation='relu'))
  return model

def prepare_lstm(train, test, filter_fn=None):
  def prepare_numeric(data):
    X = []
    y = []
    names = []
    for card in data:
        X.append(np.concatenate((card.types, [card.power, card.toughness, card.loyalty], card.colors)))
        y.append(card.cost)
        names.append(card.name)
    return X, y, names

  def prepare_text(data, tokenizer):
    corpus = [t[2] for t in data]
    tokens = tokenizer.texts_to_sequences(corpus)
    text = sequence.pad_sequences(tokens, maxlen=MAX_LEN)
    return text

  def create_trained_tokenizer(data):
    tokenizer = MTGTokenizer(nb_words=VOCAB_SIZE, filters=None, lower=True, split=" ")
    corpus = [t[2] for t in data]
    tokenizer.fit_on_texts(corpus)
    return tokenizer

  if filter_fn:
    train = filter(filter_fn, train)
    test = filter(filter_fn, test)

  tokenizer = create_trained_tokenizer(train)
  X_train_text = prepare_text(train, tokenizer)
  X_test_text = prepare_text(test, tokenizer)
  X_train_numeric, y_train, _ = prepare_numeric(train)
  X_test_numeric, y_test, y_test_names = prepare_numeric(test)

  #Combine text+numeric data
  X_train = map(np.asarray, [X_train_text, X_train_numeric])
  X_test = map(np.asarray, [X_test_text, X_test_numeric])
  return X_train, np.asarray(y_train), X_test, np.asarray(y_test), y_test_names

def lstm_mlp(X_train, y_train, X_test, y_test, pretrain=None):
  print "lstm_mlp"
  model = build_full_model(X_train[1][0].shape, pretrain)
  print "Compiling..."
  model.compile(loss='msle', optimizer='adam')
  print "Fitting..."
  model.fit(X_train, y_train, batch_size=256, nb_epoch=250, validation_split=.1)
  model.save_weights("weights_1.model", overwrite=True)

def lstm_pretrain(X_train, y_train):
  print "lstm_pretrain"
  y_train_sum = np.sum(y_train, axis=1)
  y_test_sum = np.sum(y_train, axis=1)
  model = build_pretrain_model()
  print "Compiling..."
  model.compile(loss='msle', optimizer='adam')
  model.fit(X_train, y_train_sum, batch_size=256, nb_epoch=50, validation_split=0.1)
  model.save_weights("pretrain_1.model", overwrite=True)
  return model

def load_pretrain():
  model = build_pretrain_model()
  model.load_weights("pretrain_1.model")
  return model

def make_predictions(X_test, y_test, y_names):
  print "make_predictions"
  model = build_full_model(X_test[1][0].shape)
  print "Compiling..."
  model.compile(loss='msle', optimizer='adam')
  model.load_weights("weights_1.model")

  print "Predicting..."
  results = model.predict(X_test)
  with open("output.txt",'w') as f:
    for result, correct, x_test, y_name in zip(results, y_test, X_test[1], y_names):
        print >> f, mana_str(result), "\t", mana_str(correct), "\t", y_name.encode('utf-8').strip()
  return model

def mana_str(cost):
  cost = round_cost(cost)
  cost_str  = "W"*cost[0]
  cost_str += "U"*cost[1]
  cost_str += "B"*cost[2]
  cost_str += "R"*cost[3]
  cost_str += "G"*cost[4]
  if cost[5] > 0:
    cost_str = str(cost[5])+cost_str
  return cost_str

def round_cost(cost):
  return map(int,map(round, cost))

def filter_data(X, y, filter_fn):
    X_out = []
    y_out = []
    for xi, yi in zip(X, y):
        print xi
        print yi
        if filter_fn(xi):
            X_out.append(xi)
            y_out.append(yi)
    return X_out, y_out

def main():
  train, test = load_set_data(after='RAV', ignore=['PLC', 'FUT'])
  remove_creatures = lambda x: x.types[0] == 0
  #X_pretrain, y_pretrain, _, _, _ = prepare_lstm(train, test, remove_creatures)
  #pretrain = lstm_pretrain(X_pretrain[0], y_pretrain)
  #pretrain = load_pretrain()
  X_train, y_train, X_test, y_test, y_test_names = prepare_lstm(train, test)
  lstm_mlp(X_train, y_train, X_test, y_test)
  make_predictions(X_test, y_test, y_test_names)

if __name__=="__main__":
    main()

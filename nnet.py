#!/usr/bin/env python
from mtg_data import load_card_data
from w2v_mtg import MTGTokenizer
import numpy as np
from keras.models import Graph, Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.core import Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from scipy.sparse import csc_matrix
from sklearn.preprocessing import OneHotEncoder

def build_language_model():
  model = Sequential()
  model.add(Embedding(2001, 256, mask_zero=True, input_length=75)) #vocab, size
  model.add(LSTM(256))
  model.add(Dropout(.5))
  #model.add(Dense(128, activation='relu')) 
  #model.add(Dropout(.5))
  return model

def build_numeric_model(input_shape):
  model = Sequential()
  model.add(Dense(256, input_shape=input_shape, activation='relu'))
  model.add(Dropout(.5))
  model.add(Dense(256, activation = 'relu'))
  model.add(Dropout(.5))
  return model

def build_full_model(input_shape):
  print "Building model with shape %s" % input_shape
  language_model = build_language_model()
  numeric_model = build_numeric_model(input_shape)
  model = Sequential()
  model.add(Merge([language_model, numeric_model], mode='concat', concat_axis=-1))
  model.add(Dropout(.5))
  model.add(Dense(6, activation='relu'))
  return model

def prepare_lstm(train, test, vocabulary_size=2000, max_len=75):
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
    text = sequence.pad_sequences(tokens, maxlen=max_len)
    return text

  def create_trained_tokenizer(data):
    tokenizer = MTGTokenizer(nb_words=vocabulary_size, filters=None, lower=True, split=" ")
    corpus = [t[2] for t in data]
    tokenizer.fit_on_texts(corpus)
    return tokenizer

  tokenizer = create_trained_tokenizer(train)
  X_train_text = prepare_text(train, tokenizer)
  X_test_text = prepare_text(test, tokenizer)
  X_train_numeric, y_train, _ = prepare_numeric(train)
  X_test_numeric, y_test, y_test_names = prepare_numeric(test)

  #Combine text+numeric data
  X_train = map(np.asarray, [X_train_text, X_train_numeric])
  X_test = map(np.asarray, [X_test_text, X_test_numeric])
  return X_train, np.asarray(y_train), X_test, np.asarray(y_test), y_test_names

def lstm_mlp(X_train, y_train, X_test, y_test):
  print "lstm_mlp"
  model = build_full_model(X_train[1][0].shape)
  print "Compiling..."
  model.compile(loss='msle', optimizer='adam')
  print "Fitting..."
  model.fit(X_train, y_train, batch_size=128, nb_epoch=50, validation_split = .1, show_accuracy=True)
  model.save_weights("weights_1.model", overwrite=True)

def make_predictions(X_test, y_test, y_names):
  print "make_predictions"
  model = build_full_model(X_test[1][0].shape)
  print "Compiling..."
  model.compile(loss='msle', optimizer='adam')
  model.load_weights("weights_1.model")

  print "Predicting..."
  results = model.predict(X_test)
  for result, correct, x_test, y_name in zip(results, y_test, X_test[1], y_names):
      print mana_str(result), "\t", mana_str(correct), "\t", y_name

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

def main():
  train, test = load_card_data()
  X_train, y_train, X_test, y_test, y_test_names = prepare_lstm(train, test)
  lstm_mlp(X_train, y_train, X_test, y_test)
  make_predictions(X_test, y_test, y_test_names)

if __name__=='__main__':
  main()

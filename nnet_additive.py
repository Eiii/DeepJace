#!/usr/bin/env python
from mtg_data import load_card_data, load_set_data
from w2v_mtg import MTGTokenizer
import numpy as np
from keras.models import Graph, Sequential
from keras.layers import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.core import Merge, Flatten, TimeDistributedMerge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers.convolutional import AveragePooling1D, Convolution1D, MaxPooling1D
import mtg_data
from sklearn.preprocessing import OneHotEncoder
import theano
import pickle
import theano.tensor as T
from naive_classifier import train_randomforest

VOCAB_SIZE=2000
MAX_LEN=40
DROPOUT=0.5
EMBEDDING_SIZE=256
BATCH_SIZE=256
CMC_PENALTY=5.0

def build_language_model():
  model = Sequential()
  model.add(Embedding(VOCAB_SIZE+1, EMBEDDING_SIZE, mask_zero=True, input_length=MAX_LEN)) #vocab, size
  model.add(LSTM(256, input_shape=(EMBEDDING_SIZE, MAX_LEN), dropout_W=DROPOUT, dropout_U=DROPOUT, return_sequences=False))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(2, activation='relu', init='he_normal')) 
  return model



def build_numeric_model(input_shape):
  model = Sequential()
  model.add(Dense(256, input_shape=input_shape, activation='relu', init='he_normal'))
  model.add(Dropout(DROPOUT))
  model.add(Dense(256, activation= 'relu'))
  model.add(Dense(2, activation = 'relu', init='he_normal'))
  return model

def build_full_model(input_shape, pretrain_language=None, pretrain_numeric=None):
  if pretrain_language is None:
    language_model = build_language_model()
  else:
    language_model = pretrain_language
    language_model.layers.pop()
  if pretrain_numeric is None:
    numeric_model = build_numeric_model(input_shape)
  else:
    numeric_model = pretrain_numeric
    numeric_model.layers.pop()
  model = Sequential()
  model.add(Merge([language_model, numeric_model], mode='sum', concat_axis=-1))
  model.add(Activation('relu'))
  return model

def prepare_lstm_data(train, test, filter_fn=None):
  def prepare_numeric(data):
    X = []
    y = []
    names = []
    for card in data:
        X.append(np.concatenate((card.types, card.colors, [card.power, card.toughness, card.loyalty])))
        y.append(card.cost)
        names.append(card.name)
    return np.asarray(X), np.asarray(y), np.asarray(names)

  if filter_fn:
    train = filter(filter_fn, train)
    test = filter(filter_fn, test)
   
  X_train_text = [ card.tokens for card in train ]
  X_test_text = [ card.tokens for card in test ] 
  X_train_text = sequence.pad_sequences(np.asarray(X_train_text), MAX_LEN)
  X_test_text = sequence.pad_sequences(X_test_text, MAX_LEN)
  X_train_numeric, y_train, _ = prepare_numeric(train)
  X_test_numeric, y_test, y_test_names = prepare_numeric(test)

  #Combine text+numeric data
  X_train = map(np.asarray, [X_train_text, X_train_numeric])
  X_test = map(np.asarray, [X_test_text, X_test_numeric])
  return X_train, np.asarray(y_train), X_test, np.asarray(y_test), y_test_names

def lstm_mlp(X_train, y_train, X_test, y_test, pretrain_lstm=None, pretrain_mlp=None, previous_model = None):
  print "lstm_mlp"
  model = build_full_model(X_train[1][0].shape, pretrain_lstm, pretrain_mlp)
  print "Compiling..."
  model.compile(loss=custom_loss, optimizer='adam')
  if previous_model != None:
    model.load_weights("weights_1.model")
  print "Fitting..."
  model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=100, validation_split=.1, show_accuracy=True)
  model.save_weights("weights_1.model", overwrite=True)

def load_previous_model(X_train, y_train):
  print "load_previous_model"
  model = build_full_model(X_train[1][0].shape)
  print "Compiling..."
  model.compile(loss=custom_loss, optimizer='adam')
  model.load_weights("weights_1.model")
  return model

def make_predictions(X_test, y_test, y_names):
  print "make_predictions"
  pred = []
  model = build_full_model(X_test[1][0].shape)
  #model = build_language_model()
  print "Compiling..."
  model.compile(loss=custom_loss, optimizer='adam')
  model.load_weights("weights_1.model")

  print "Predicting..."
  results = model.predict(X_test)
  with open("output.txt",'w') as f:
    for result, correct, x_test, y_name in zip(results, y_test, X_test[1], y_names):
        print >> f, mana_str(result), "\t", mana_str(correct), "\t", y_name.encode('utf-8').strip()
        pred.append((result.tolist(), correct.tolist()))
  return pred

def mana_str(cost):
  #cost = round_cost(cost)
  color = cost[0]
  cless = cost[1]
  cmc = cost[0]+cost[1]
  cost_str = "%s %s %s" % (color, cless, cmc)
  return cost_str

def round_cost(cost):
  return map(int,map(round, cost))

def filter_data(X, y, filter_fn):
    X_out, y_out  = [], []
    for xi, yi in zip(X, y):
        print xi, "\n", yi
        if filter_fn(xi):
            X_out.append(xi)
            y_out.append(yi)
    return X_out, y_out

def custom_loss(y_true, y_pred):
  epsilon = 0.001
  first_log = T.log(T.clip(y_pred, 0.001, np.inf) + 1.)
  second_log = T.log(T.clip(y_true, 0.001, np.inf) + 1.)
  first_sum = T.log(T.sum(T.clip(y_pred, 0.001, np.inf))+1)
  second_sum = T.log(T.sum(T.clip(y_true, 0.001, np.inf))+1)
  return T.mean(T.square(first_log-second_log), axis=-1) + CMC_PENALTY*T.square(first_sum-second_sum)

def main():
  train, test = load_set_data(after="MRD", ignore=["AVR", "ISD", "DKA"])#, only_types=mtg_data.SPELL_TYPES)
  remove_creatures = lambda x: x.types[0] == 0
  X_train, y_train, X_test, y_test, y_test_names = prepare_lstm_data(train, test)
   
  #pre = load_previous_model(X_train, y_train)
  lstm_mlp(X_train, y_train, X_test, y_test, previous_model=None)#,lstm, mlp)
  result = make_predictions(X_test, y_test, y_test_names)
  pickle.dump(result, open('output.p', 'wb'))
  train_randomforest(train, test, n_estimators=20, cpus=4)

if __name__=="__main__":
    main()

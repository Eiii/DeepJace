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
import theano.tensor as T

VOCAB_SIZE = 2000
MAX_LEN = 40
DROPOUT = 0.1
EMBEDDING_SIZE = 256
batch_size=256
FILTER_LENGTH=3
pool_length=2
def build_language_model():



  model = Sequential()
  model.add(Embedding(VOCAB_SIZE+1, EMBEDDING_SIZE, mask_zero=True, input_length=MAX_LEN)) #vocab, size
  model.add(Dropout(DROPOUT))
  """  
  model.add(Convolution1D(nb_filter=128,
                            filter_length=FILTER_LENGTH,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
  model.add(Dropout(DROPOUT))
  model.add(MaxPooling1D(pool_length=pool_length, stride=1))
  model.add(Convolution1D(nb_filter=128 * 2,
                            filter_length=FILTER_LENGTH,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
  model.add(MaxPooling1D(pool_length=pool_length, stride=1))
  model.add(Flatten())
  """
  model.add(LSTM(256, input_shape=(EMBEDDING_SIZE, MAX_LEN), dropout_W=DROPOUT, dropout_U=DROPOUT, return_sequences=False))
  model.add(Dense(256, activation='relu'))
  model.add(Dense(2, activation='relu', init='he_normal')) 

#  model.add(Dense(252, activation='relu', init='he_normal'))
#  model.add(Dropout(DROPOUT))
  return model

def cmc_loss(y_true, y_pred):
  epsilon = 0.001
  first_sum = T.log(T.sum(T.clip(y_pred, 0.001, np.inf)))
  second_sum = T.log(T.sum(T.clip(y_true, 0.001, np.inf)))
  return T.square(first_sum - second_sum)

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
  return model

def build_pretrain_model(base_model, dense_size=2):
  #base_model.add(Dense(dense_size, activation='relu'))
  return base_model

def prepare_lstm_data(train, test, filter_fn=None):
  def prepare_numeric(data):
    X = []
    y = []
    names = []
    for card in data:
        X.append(np.concatenate((card.types, [card.power, card.toughness, card.loyalty], card.colors)))
        y.append(card.cost)
        names.append(card.name)
    return np.asarray(X), np.asarray(y), np.asarray(names)

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

def lstm_mlp(X_train, y_train, X_test, y_test, pretrain_lstm=None, pretrain_mlp=None, previous_model = None):
  print "lstm_mlp"
  print X_train
  model = build_full_model(X_train[1][0].shape, pretrain_lstm, pretrain_mlp)
  print "Compiling..."
  model.compile(loss='msle', optimizer='rmsprop')
  if previous_model != None:
    model.load_weights("weights_1.model")
  print "Fitting..."
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=200, validation_split=.1, show_accuracy=True)
  model.save_weights("weights_1.model", overwrite=True)

def lstm_pretrain(X_train, y_train):
  print "lstm_pretrain"
  #y_train_sum = np.sum(y_train, axis=1)
  #y_test_sum = np.sum(y_train, axis=1)
  base = build_language_model()
  model = build_pretrain_model(base, 2)
  print "Compiling..."
  model.compile(loss='msle', optimizer='adam')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=75, validation_split=0.1)
  model.save_weights("pretrain_lstm.model", overwrite=True)
  return model

def mlp_pretrain(X_train, y_train):
  print "mlp+pretrain"
  base = build_numeric_model(X_train[0].shape)
  model =  build_pretrain_model(base)
  print "Compiling..."
  model.compile(loss='msle', optimizer='adam')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=75, validation_split=0.1)
  model.save_weights("pretrain_numeric.model", overwrite=True)
  

def load_lstm_pretrain():
  base = build_language_model()
  model = build_pretrain_model(base, 2)
  model.load_weights("pretrain_lstm.model")
  return model

def load_mlp_pretrain(input_shape):
  base = build_numeric_model(input_shape)
  model = build_pretrain_model(base)
  model.load_weights("pretrain_numeric.model")
  return model

def load_previous_model(X_train, y_train):
  print "make_predictions"
  model = build_full_model(X_train[1][0].shape)
  print "Compiling..."
  model.compile(loss='msle', optimizer='adam')
  model.load_weights("weights_1.model")
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
  cost_str = "C" * cost[0]
  if cost[1] > 0:
    cost_str += str(cost[1])
  return cost_str
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

def custom_loss(y_true, y_pred):
  epsilon = 0.001
  first_log = T.log(T.clip(y_pred, 0.001, np.inf) + 1.)
  second_log = T.log(T.clip(y_true, 0.001, np.inf) + 1.)
  first_sum = T.log(T.sum(T.clip(y_pred, 0.001, np.inf)))
  second_sum = T.log(T.sum(T.clip(y_true, 0.001, np.inf)))
  return T.mean(T.square(first_log - second_log), axis=-1)+T.square(first_sum - second_sum)

def main():
  train, test = load_set_data(after="MRD")#, only_types=mtg_data.SPELL_TYPES)

  remove_creatures = lambda x: x.types[0] == 0
  #train = train[:,8:]
  #test = test[:,8:]

  X_pretrain, y_pretrain, _, _, _ = prepare_lstm_data(train, test)#, remove_creatures)
  X_pretrain_mlp, y_pretrain_mlp, _, _, _ = prepare_lstm_data(train, test)
  #X_pretrain_mlp[1]  = X_pretrain_mlp[1][:,8:]
  #np.delete(X_pretrain_mlp[1], -2, axis=1)

  #pretrain_lstm = lstm_pretrain(X_pretrain[0], y_pretrain)
  #pretrain_mlp = mlp_pretrain(X_pretrain_mlp[1], y_pretrain_mlp) 

  #lstm = load_lstm_pretrain()
  #mlp = load_mlp_pretrain(X_pretrain_mlp[1][1].shape)
 
  X_train, y_train, X_test, y_test, y_test_names = prepare_lstm_data(train, test)
  #X_train[1] = X_train[1][:,8:]
  #X_test[1] = X_test[1][:,8:]
  X_train[1] = X_train[1][:,:-6]
  X_test[1] = X_test[1][:,:-6]
  y_train_n = []
  y_test_n = [] 
  
  #y_train = np.sum(y_train, axis=1)
  #y_test = np.sum(y_test, axis=1)
  #pre = load_previous_model(X_train, y_train)
  lstm_mlp(X_train, y_train, X_test, y_test, previous_model=None)#,lstm, mlp)
  make_predictions(X_test, y_test, y_test_names)

if __name__=="__main__":
    main()

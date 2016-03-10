#!/usr/bin/env python
from mtg_data import load_card_data
from w2v_mtg import MTGTokenizer
import numpy as np

train, test = load_card_data()

print len(train), len(test)
print train[0]


def prepare_classic_implementation():
    from keras.preprocessing.text import Tokenizer
    import numpy as np
    from scipy.sparse import csc_matrix
    from sklearn.preprocessing import OneHotEncoder

    vocabulary_size = 2000
    #keep commas and colons

    train, test = load_card_data()
    corpus = [t[2] for t in train]
    test_corpus = [t[2] for t in test]

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    """
    prepare text training data
    """
    count_vect = CountVectorizer(max_features=None)
    X_train_counts = count_vect.fit_transform(corpus)
    X_test_counts = count_vect.transform(test_corpus)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_test_tf = tf_transformer.transform(X_test_counts)
    
    X_names = []
    X_train = []
    y_train = []
    for (name, cost, text, types, power, toughness, loyalty, colors), token_text in zip(train, X_train_tf):
        X_names.append(name)
        features = np.concatenate((token_text.toarray().flatten(), types, [power, toughness, loyalty], colors))
        X_train.append(features)
        y_train.append(cost)

    X_test = []
    y_test = []
    X_test_names = []
    for (name, cost, text, types, power, toughness, loyalty, colors), token_text in zip(test, X_test_tf):
        X_test_names.append(name)
        features = np.concatenate((token_text.toarray().flatten(), types, [power, toughness, loyalty], colors))
        X_test.append(features)
        y_test.append(cost)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import cross_validation
    rf= RandomForestRegressor(n_estimators=100, n_jobs=4)
    rf.fit(X_train, y_train)
    print "train score", rf.score(X_train, y_train)
    print "test score", rf.score(X_test, y_test)

    print "train mse", np.mean((rf.predict(X_train) - y_train) ** 2)
    print "test mse", np.mean((rf.predict(X_test) - y_test) ** 2)
    print ""
    print "mlp accuracy/loss"
    print "Name:", X_test_names[0]
    mlp(X_train, X_train, y_train, X_test)

def prepare_lstm():
    from keras.preprocessing.text import Tokenizer
    import numpy as np
    from scipy.sparse import csc_matrix
    from sklearn.preprocessing import OneHotEncoder
    from keras.preprocessing import sequence

    vocabulary_size = 2000
    #keep commas and colons


    train, test = load_card_data()
    corpus = [t[2] for t in train]
    test_corpus = [t[2] for t in test]

    tokenizer = MTGTokenizer(nb_words=vocabulary_size, filters=None, lower=True, split=" ")
    tokenizer.fit_on_texts(corpus)
    train_tokens = tokenizer.texts_to_sequences(corpus)
    test_tokens = tokenizer.texts_to_sequences(test_corpus)
    maxlen=75
    X_train_text = train_tokens
    X_test_text = test_tokens
    X_train_text = sequence.pad_sequences(X_train_text, maxlen=maxlen)
    X_test_text = sequence.pad_sequences(X_test_text, maxlen=maxlen)

    X_train_numeric = []
    y_train = []
    y_names = []
    for (name, cost, text, types, power, toughness, loyalty, colors) in train:
        y_names.append(name)
        X_train_numeric.append(np.concatenate((types, [power, toughness, loyalty], colors)))
        y_train.append(cost)
    X_test_numeric = []
    X_test_names = []
    y_test = []
    for (name, cost, text, types, power, toughness, loyalty, colors) in test:
        X_test_names.append(name)
        X_test_numeric.append(np.concatenate((types, [power, toughness, loyalty], colors)))
        y_test.append(cost)

    #lstm_mlp(np.asarray(X_train_text), np.asarray(X_train_numeric), np.asarray(y_train), np.asarray(X_test_text), np.asarray(X_test_numeric), y_test)
    make_predictions(np.asarray(X_test_text), np.asarray(X_test_numeric), y_test)

    

def lstm_mlp(X_train_text, X_train_other, y_train, X_test_text=[], X_test_other=[], y_test=[]):
    from keras.models import Graph, Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers.core import Merge
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM
    print X_train_text.shape, X_train_other.shape, y_train.shape
    maxlen = 50
    #make sure the vocab size is the same as our earlier stuff
    print "shapes"
    print X_train_text.shape
    print X_train_text[0].shape
    print y_train.shape
    language_model = Sequential()
    language_model.add(Embedding(2001, 256, mask_zero=True, input_length=75)) #vocab, size
    language_model.add(LSTM(256))
    language_model.add(Dropout(.5))
    #language_model.add(Dense(128, activation='relu')) 
    #language_model.add(Dropout(.5))

    print "merge models"

    numerical_model = Sequential()
    numerical_model.add(Dense(256, input_shape = X_train_other[0].shape, activation='relu'))
    numerical_model.add(Dropout(.5))
    numerical_model.add(Dense(256, activation = 'relu'))
    numerical_model.add(Dropout(.5))

    model = Sequential()
    model.add(Merge([language_model, numerical_model], mode='concat', concat_axis=-1))
    model.add(Dropout(.5))
    model.add(Dense(6, activation='relu'))
    model.compile(loss='msle', optimizer='adam')

    model.fit([X_train_text, X_train_other], y_train, batch_size=128, nb_epoch=50, validation_split = .1, show_accuracy=True)
    print model.predict([X_test_text[0:3], X_test_other[0:3]]), y_test[0:3]


    model.save_weights("weights_1.model", overwrite=True)

def make_predictions(X_test_text, X_test_other, y_test):
    from keras.models import Graph, Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.layers.core import Merge
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM
    language_model = Sequential()
    language_model.add(Embedding(2001, 256, mask_zero=True, input_length=75)) #vocab, size
    language_model.add(LSTM(256))
    language_model.add(Dropout(.5))
    #language_model.add(Dense(256, activation='relu'))
    #language_model.add(Dropout(.5))


    numerical_model = Sequential()
    numerical_model.add(Dense(256, input_shape = X_test_other[0].shape, activation='relu'))
    numerical_model.add(Dropout(.5))
    numerical_model.add(Dense(256, activation='relu'))
    numerical_model.add(Dropout(.5))

    model = Sequential()
    model.add(Merge([language_model, numerical_model], mode='concat', concat_axis=-1))
    model.add(Dropout(.5))
    model.add(Dense(6, activation='relu'))
    model.compile(loss='msle', optimizer='adam')
    model.load_weights("weights_1.model")

    results = model.predict([X_test_text, X_test_other])
    for result, correct, x_test in zip(results, y_test, X_test_other):
        print ",".join([str(r) for r in result]) + "\t" + ",".join([str(r) for r in correct])
    
    

def mlp(X_train, y_train, X_test, y_test):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD

    model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
    print X_train[0].shape
    model.add(Dense(512, input_dim=X_train[0].shape[0], init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(6, init='uniform'))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam')

    model.fit(X_train, y_train,
              nb_epoch=200,
              batch_size=32,
              show_accuracy=True,
              validation_split=.1)
    score = model.evaluate(X_test, y_test, batch_size=16)
    #print model.predict[X_test_text[0:1], X_test_other], y_test[0])
    print "mlp score", score
            
import theano
import theano.tensor as T

epsilon = 1.0e-7
def mtg_objective(y_true, y_pred):
    ''' mse adjusted for cmc and colors -- There is a weird bitop errro when you try and combine any of these results. might need to vectorize it all.'''
    """
    first_log_nc = T.log((T.sum(T.clip(y_pred[:-1], epsilon, np.inf) + 1.)
    second_log_nc = T.log((T.sum(T.clip(y_true[:-1], epsilon, np.inf) + 1.) 

    first_log_cmc = T.log(T.sum(T.clip(y_pred, epsilon, np.inf) + 1.))
    second_log_cmc = T.log(T.sum(T.clip(y_true, epsilon, np.inf) + 1.))
    
    first_log_c = T.log(T.clip(y_pred[:-1], epsilon, 1) + 1)
    second_log_c = T.log(T.clip(y_true[:-1], epsilon, 1) + 1)

    first_tot = T.log(T.clip(y_pred, epsilon, np.inf) + 1.)
    second_tot = T.log(T.clip(y_true, epsilon, np.inf) + 1.)

    result = T.square(first_log_nc - second_log_nc)
    return result
    return T.square(first_log_nc - second_log_nc) + T.square(T.sum(first_log_c - second_log_c))
    """
    return 0
    #return T.square(T.sum(first_log_c - second_log_c))/3 
    #return T.square(T.sum(first_log_c - second_log_c))/ 3 + T.square(first_log_cmc - second_log_cmc)/3 + T.square(first_log_nc - second_log_nc)/3
#    return first_tot - second_tot + first_log_cmc - second_log_cmc + first_log_nc - second_log_nc + first_log_c - second_log_c
#    return T.mean( T.square(first_log_cmc - second_log_cmc)/3 +  T.square(first_log_c - second_log_c)/3 +  T.square(first_log_nc - second_log_nc)/3, axis=-1)

    

#prepare_onehot_vectors()
prepare_lstm()

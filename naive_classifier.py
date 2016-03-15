#!/usr/bin/env python
from mtg_data import load_card_data, load_set_data
from w2v_mtg import MTGTokenizer
import pickle
import numpy as np

def train_randomforest(train, test, n_estimators=10, cpus=4):
    import numpy as np
    from scipy.sparse import csc_matrix
    from sklearn.preprocessing import OneHotEncoder

    vocabulary_size = 2000
    #keep commas and colons

    corpus = [t.text for t in train]
    test_corpus = [t.text for t in test]

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
    for card, token_text in zip(train, X_train_tf):
        X_names.append(card.name)
        features = np.concatenate((token_text.toarray().flatten(), card.types, [card.power, card.toughness, card.loyalty], card.colors))
        X_train.append(features)
        y_train.append(card.cost)

    X_test = []
    y_test = []
    X_test_names = []
    for card, token_text in zip(test, X_test_tf):
        X_test_names.append(card.name)
        features = np.concatenate((token_text.toarray().flatten(), card.types, [card.power, card.toughness, card.loyalty], card.colors))
        X_test.append(features)
        y_test.append(card.cost)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import cross_validation
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=cpus)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    print y_pred.shape, y_train.shape
    print "naive train loss", np.mean(custom_loss(y_train, y_pred))
    y_pred = rf.predict(X_test)
    print "naive test loss", np.mean(custom_loss(y_test, y_pred))
    result = print_predictions(y_pred, y_test, X_test_names)
    print "saving to output.naive.txt and output.naive.p"
    pickle.dump(result, open('output.naive.p', 'wb'))


def mana_str(cost):
  #cost = round_cost(cost)
  color = cost[0]
  cless = cost[1]
  cmc = cost[0]+cost[1]
  cost_str = "%s %s %s" % (color, cless, cmc)
  return cost_str

def print_predictions(results, y_test, y_names):
    pred = []
    with open("output.naive.txt",'w') as f:
        for result, correct, y_name in zip(results, y_test, y_names):
            print >> f, mana_str(result), "\t", mana_str(correct), "\t", y_name.encode('utf-8').strip()
            pred.append((result.tolist(), correct.tolist()))
        return pred

def custom_loss(y_true, y_pred):
  epsilon = 0.001
  CMC_PENALTY = 5.0
  first_log = np.log(np.clip(y_pred, 0.001, np.inf) + 1.)
  second_log = np.log(np.clip(y_true, 0.001, np.inf) + 1.)
  first_sum = np.log(np.sum(np.clip(y_pred, 0.001, np.inf))+1)
  second_sum = np.log(np.sum(np.clip(y_true, 0.001, np.inf))+1)
  return np.mean(np.square(first_log-second_log), axis=-1) + CMC_PENALTY*np.square(first_sum-second_sum)

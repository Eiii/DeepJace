#!/usr/bin/env python
import json
import bz2 as bz
import nltk
import enchant
import time
from nltk.tokenize import word_tokenize
import numpy as np
import sys
import string
import random

d = None
def prepare_reddit(filename, subreddit):
    data = []
    by_sub = {}
    """
    """
    if filename[-3:] == "bz2":
        fd = bz.BZ2File(filename, 'r')
    else:
        fd = open(filename, 'r')
    for line in fd:
        info = json.loads(line.strip())
        
    #    if info["subreddit"] not in by_sub:
    #        by_sub[info["subreddit"]] = []
        try:    
            body = info["body"].encode("utf-8").lower().replace("\n", " ").strip()
            body = body.translate(None, string.punctuation)
            link_id = str(info["link_id"].encode("utf-8"))
            if str(body) == "" or str(body) == "deleted":
                continue
            score = int(str(info["score"]).decode("utf-8"))
            if info["subreddit"] == subreddit:
                print str(body) + "," + str(score) + "," + str(link_id)
#            by_sub[info["subreddit"]].append((body, int(str(info["score"]).decode("utf-8"))))
        except Exception as e:
            pass
    return
    sorted(by_sub["nba"], key=lambda x : x[1], reverse=True)
     
    X_train, y_train = zip(*by_sub["nba"])
    sent_tokens = [word_tokenize(x) for x in X_train]
    tokens = [item for sublist in sent_tokens for item in sublist]
    global d
    d = train_spellchecker(tokens)
    import multiprocessing as mp
    
    p = mp.Pool(16)
    start = time.time()
    correct_sentence(sent_tokens[0])
    correct_sents = p.map(correct_sentence,  sent_tokens )
    stop = time.time()
    print >> sys.stderr, "total time:", stop - start
    print >> sys.stderr, "time per sentence:", (stop - start) / len(correct_sents)
    X_train = zip(correct_sents, y_train)
    save_cleaned_data(X_train)

def save_cleaned_data(X):
    for x, y in X:
        print " ".join(x) + "," + str(int(y))
    

def byteify(input):
    if isinstance(input, dict):
        return {byteify(key):byteify(value) for key,value in input.iteritems()}
    elif isinstance(input, list):
        return [byteify(element) for element in input]
    elif isinstance(input, unicode):
        return input.encode('utf-8')
    else:
        return input
        
def train_spellchecker(tokenized_words_untouched):
    words_not_vectorized = set()
    all_words_untouched = set(tokenized_words_untouched)

    """
    Want to do this for all words in our corpus
    """
    print >> sys.stderr, 'applying frequency distribution to original text'
    word_freq = nltk.FreqDist(tokenized_words_untouched) 

    for eachword7 in all_words_untouched:
        if word_freq[eachword7] < 2: #we choose 2 because a word is rarely mispelled incorrectly twice in the same way
            words_not_vectorized.add(eachword7)

    words_that_are_common = all_words_untouched - words_not_vectorized

    print >> sys.stderr, 'creating personal spelling dictionary'
    with open ('listofspelledwords.txt','w+') as listofspelledwords:
        for eachword12 in words_that_are_common: #add to dictionary for spelling corrector
            listofspelledwords.write(eachword12+'\n')

    del words_that_are_common
    return enchant.DictWithPWL("en_US","listofspelledwords.txt")
    

#pass in a sentence and a dictionary
def correct_sentence(tokenized_words_untouched):
    global d
    spelled_tokenized_words_untouched =[]
    number_of_corrected_spelling_errors = 0 
    start_time = time.time()
    """
    Want to do this on a sentence by sentence basis
    """
    for eachword13 in tokenized_words_untouched:
        if d.check(eachword13): #word spelled correctly
            spelled_tokenized_words_untouched.append(eachword13)
        else:  #word not spelled correctly
            try:
                spelled_tokenized_words_untouched.append(d.suggest(eachword13)[0])
            # print 'changed '+eachword13+' to '+(d.suggest(eachword13)[0])
                number_of_corrected_spelling_errors = number_of_corrected_spelling_errors +1
            except IndexError:
                spelled_tokenized_words_untouched.append(eachword13)
    del number_of_corrected_spelling_errors

    return spelled_tokenized_words_untouched

def train(X, y):
    from keras.layers.embeddings import Embedding
    from keras.layers.recurrent import LSTM, GRU, SimpleRNN
    from keras.layers.core  import Dense
    from keras.models import Sequential
    from keras.layers.core import Dropout
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from math import e
    vocab = 10000
    tokenizer = Tokenizer(nb_words=vocab)
    tokenizer.fit_on_texts(X)
    X = tokenizer.texts_to_sequences(X)
    """
    index_word =  {v: k for k, v in tokenizer.word_index.items()}
    for i in range(1, 10001):
        print str(i) + "," + index_word[i]

    return
    """
    maxlen= 50
    X1 = []
    y1 = [] 
    for thing,target in zip(X, y):
        if len(thing) != 0:
            X1.append(thing)
            y1.append(target)
            
    X = X1
    y = y1
    KERAS = False
    if KERAS:
        X = pad_sequences(X, maxlen=maxlen)

    from random import shuffle
    xy = zip(X, y)
    shuffle(xy)
    X_s, y_s = zip(*xy)
    X_train, y_train, X_test, y_test = X_s[:-1000], y_s[:-1000], X_s[-1000:], y_s[-1000:]
    embedding_size = 256
    dropout = .3
    batch_size = 256
    recurrent_gate_size = 512
    
    """
    model = Sequential()
    model.add(Embedding(vocab, embedding_size, mask_zero=True))
    model.add(Dropout(dropout))
    model.add(LSTM(recurrent_gate_size))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    print "building model..."
    model.compile(loss="msle", optimizer="rmsprop")
    print "fitting model"
    #model.load_weights("mymodel")
    model.fit(np.asarray(X_train), np.asarray(y_train), nb_epoch=30, verbose=1, batch_size=batch_size, validation_data=(np.asarray(X_test), np.asarray(y_test)))
   
    model.save_weights("mymodel") 
    """
    from passage.preprocessing import Tokenizer, LenFilter
    from passage.layers import Embedding, GatedRecurrent, Dense, OneHot, LstmRecurrent
    from passage.models import RNN
    from passage.utils import save, load
    from passage.iterators import Padded

    layers = [
    #    OneHot(n_features=5),
        Embedding(size=embedding_size,n_features=vocab),
    #    GatedRecurrent(size=recurrent_gate_size, seq_output=True, p_drop=dropout),
    #    LstmRecurrent(size=recurrent_gate_size, p_drop=dropout),
        GatedRecurrent(size=recurrent_gate_size, p_drop=dropout),
        Dense(size=8, activation='softmax', p_drop=dropout)
    ]

    print >> sys.stderr, "learning model"
    model_iterator = Padded()
    model = load("mymodel.final.pkl")
    #model = RNN(layers=layers, cost='CategoricalCrossEntropy', verbose=2, updater="Adam")
    filter = LenFilter(max_len=maxlen)
    model.fit(np.asarray(X_train), np.asarray(y_train), batch_size=batch_size, n_epochs=1000, path="mymodel.pkl", snapshot_freq=49, len_filter=filter)
    save(model, "mymodel.final.pkl")
#    print "test cost"
#    print model._cost(np.asarray(X_test), np.asarray(y_test)) 
    print "test accuracy"
    passage_batch_predict(np.asarray(X_train), np.asarray(y_train), model)

    exit = False
    print "enter a sentence"
    while not exit:
        text = raw_input()
        if text == "exit":
            break
        else:
            tokens = tokenizer.texts_to_sequences([text])
            if len(tokens) == 0:
                print "Sentence too strange, try again"
                continue
            if KERAS:
                tokens = pad_sequences(tokens, maxlen=maxlen)
            prediction = np.argmax(model.predict(tokens)[0])
            try:
                print e ** (prediction- 2)
            except Exception:
                pass


def passage_batch_predict(data, labels, model, batch_size=128):
    correct, incorrect = 0,0
    abs_error = 0
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        batch_labels = data[i:i+batch_size]
        for idx, prediction in enumerate(model.predict(batch_data)):
            print np.argmax(prediction), prediction, labels[idx]
            error = abs(np.argmax(prediction) - labels[idx])
            abs_error += error
    print "absolute error"
    print float(abs_error)/len(data)

def predict(sentence, tokenizer, maxlen, model):
    tokens = tokenizer.texts_to_sequences([sentence])
    tokens = pad_sequences(tokens, maxlen=maxlen)
    model.predict(tokens)

def print_y_stats(y):
    from scipy import mean, median, std
    print mean(y), median(y), std(y)


def generate():
    pass

def load_clean(filename):
    X_train = []
    y_train = []
    by_ys = { k : [] for k in range(-2, 6)}
    print by_ys
    from math import log
    for line in open(filename, 'r'):
        x, y = line.strip().split(",")
        y = int(y)
        if y < 0:
            y = round(log(abs(y)) * -1)
        elif y == 0:
            y = 0
        else:
            y = round(log(y))
        if y < -2: y = -2
        if y > 5: y = 5
        by_ys[y].append(x) 


    """
    do sampling.... say 10k from each
    """ 
    for k in by_ys:
        X_train += random.sample(by_ys[k], 20000)
        y_train += [k+2] * 20000
    print len(X_train), len(y_train)
    return X_train, y_train

def main():
    #for arg in sys.argv[1:]:
    #    prepare_reddit(arg, "nba")
    X, y = load_clean(sys.argv[1])
    train(X, y)
    
    pass

if __name__ == "__main__":
    main()

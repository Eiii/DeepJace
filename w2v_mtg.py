#!/usr/bin/env python
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from keras.layers.core import TimeDistributedDense,Dense, Dropout, Activation
from keras.layers.embeddings import  Embedding
from keras.layers.recurrent import SimpleRNN
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
import numpy as np

def main():
    vocabulary_size = 10000
    #keep commas and colons
    corpus = [line.strip().lower().replace("(", "").replace(")", "").replace(".", "").split() for line in open('corpus.txt', 'r')]
    tokenizer = MyTokenizer(nb_words=vocabulary_size, filters=None, lower=True, split=" ")
    tokenizer.fit_on_texts(corpus)
    tokens = tokenizer.texts_to_sequences(corpus)

    X_train = []
    y_train = []
    for token in tokens:
        couples, labels = skipgrams(token, vocabulary_size) 
            
        X_train += couples
        y_train += labels
    """
        couples is a list of 2-elements lists of int: [word_index, other_word_index].
        labels is a list of 0 and 1, where 1 indicates that other_word_index was found in the same window as word_index, and 0 indicates that other_word_index was random.
    """
    
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)  
    embedding_size = 256
    model = Sequential()
    model.add(Embedding(vocabulary_size+1, 256))
    model.add(SimpleRNN(128, return_sequences=False))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="Adam", loss="binary_crossentropy")
    model.load_weights("mtgW2V.mdl")
    embedding_weights = model.layers[0].get_weights()[0]
    
    from scipy.spatial.distance import cosine
    embedding_dict = tokenizer.word_index
    top5 = []
    for word1 in embedding_dict:
        print word1
        top5 = [] 
        bottom5 = []
        scores = []
        for word2 in embedding_dict:
            score = 1 - cosine(embedding_weights[embedding_dict[word1]], embedding_weights[embedding_dict[word2]])
            if score > min(top5):
                top5.append(word2)
            if score < max(bottom5):
                bottom5.append(word2)
        
    
    
    while True:
        try:
                word1 = raw_input("What is the first word you want to compare?")
                word2 = raw_input("What is the second word you want to compare?")
                print embedding_dict[word1], embedding_dict[word2]
                #1 - gets us similarity instead of distance
                print 1-cosine(embedding_weights[embedding_dict[word1]], embedding_weights[embedding_dict[word2]])
        except KeyError:
            pass
        except IndexError:
            pass
    
    
#    model.fit(X_train, y_train, nb_epoch=2, batch_size=1024)
    

#    model.save_weights("mtgW2V.mdl")
    

class MyTokenizer(Tokenizer):
    def __init__(self, *args, **kwargs):
        super(MyTokenizer, self).__init__(*args, **kwargs)

    def fit_on_texts(self, texts):
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            seq = text
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
        self.index_docs = {}
        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences_generator(self, texts):
        '''
            Transform each text in texts in a sequence of integers.
            Only top "nb_words" most frequent words will be taken into account.
            Only words known by the tokenizer will be taken into account.

            Yields individual sequences.
        '''
        nb_words = self.nb_words
        for text in texts:
            seq = text
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        pass
                    else:
                        vect.append(i)
            yield vect

if __name__ == "__main__":
    main()

#!/usr/bin/env python
from nnet_additive import make_predictions,  prepare_lstm_data, build_full_model
from mtg_data import load_set_data
import sys
def make_predictions(X_test, y_test, y_names):
  print "make_predictions"
  model = build_full_model(X_test[1][0].shape)
  model.compile(loss='msle', optimizer='adam')
  model.load_weights("weights_1.model")
  language = model.layers[0].layers[0]
  numerical = model.layers[0].layers[1]
  language.compile(loss='msle', optimizer='adam')
  numerical.compile(loss='msle', optimizer='adam')

  print "Predicting..."
  lp = language.predict(X_test[0])
  num_p = numerical.predict(X_test[1])
  print lp, num_p 
  for l, n, name, y in zip(lp, num_p, y_names, y_test):
    print str(name), str(y), str(l), str(n)
    
  return model

def main():
  data = load_set_data(as_dictionary=True)
  
  cardlist = []
  for line in open("cardlist.txt", 'r'):
    cardlist.append(line.strip())
  to_test = []
  for card in cardlist:
    try:
      to_test.append(data[card])
    except Exception as e:
      print e
  
  _, _, X, y, names = prepare_lstm_data([], to_test)
  make_predictions(X, y, names)

if __name__ == "__main__":
    main()

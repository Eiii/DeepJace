#!/usr/bin/env python2
from w2v_mtg import MTGTokenizer
from collections import namedtuple
import json
import random
import numpy as np
import re 

SPELL_TYPES = ['Instant', 'Enchantment', 'Sorcery', 'Enchant']
SPELLS_N_PLANESWALKERS = SPELL_TYPES + ["Planeswalker"]
CARD_DATA = 'data/AllCards.json'
SET_DATA = 'data/AllSets.json'
WHITE, BLUE, BLACK, RED, GREEN, COLORLESS = range(6)

Card = namedtuple('Card', ['name', 'cost', 'text', 'types', 'power', 'toughness', 'loyalty', 'colors', 'set', 'tokens'])

"""
load_card_data
Generates a train and test data set.
Only loads `data_pct` percent of the total data set.
The test data set will be `test_pct` percent of all the data loaded.
Returns [training data], [testing data]

X - cost spells do not function. * cost power/toughness creatures do not function.
"""
def load_card_data(test_pct=0.1, data_pct=1, seed=1337, vocab_size=2000):
  random.seed(seed)
  with open(CARD_DATA) as f:
    json_data = json.load(f).values()
  return json_data
  card_data = []
  error_cards = 0
  for j in json_data:
    if only_types is not None and 'types' in j and \
       all([str(t) not in only_types for t in j['types']]):
      continue
    try:
      c = convert_json_card(j)
      card_data.append(c)
    except Exception as e:
      error_cards += 1
  random.shuffle(card_data)
  card_data = { card[0] : card for card in card_data }
  card_data = card_data.values()
  if data_pct < 1:
    amt = int(len(card_data)*data_pct)
    card_data = card_data[:max(amt, 1)]

  test_amt = int(len(card_data)*test_pct)
  test_data = card_data[:test_amt]
  train_data = card_data[test_amt:]
  return train_data, test_data

def load_set_data(test_pct=0.1, data_pct=1, seed=1337, before=None, after=None, ignore=None, only_types=None, as_dictionary=False,vocab_size=2000):
  random.seed(seed)

  with open(SET_DATA) as f:
    json_data = json.load(f)

  set_order = order_sets(json_data)
  if before:
    legal_sets = sets_before(set_order, before)
  elif after:
    legal_sets = sets_after(set_order, after)
  else:
    legal_sets = json_data.keys()
  legal_set_types = ['core', 'expansion']

  card_data = []
  error_cards = 0
  corpus = []
  """ load corpus """
  for set_name in json_data.keys():
    for card in json_data[set_name]['cards']:
      try:
          name = card['name']
          corpus.append(convert_text(card['text'], name))
      except Exception as e:
          ''' no text '''
          pass
  tokenizer = MTGTokenizer(nb_words=vocab_size, filters=None, lower=True, split=" ")
  tokenizer.fit_on_texts(corpus)
  save_top_n(tokenizer, 100)
  
  """ tuples makin me loop twice -- load and filter card data """          
  for set_name in json_data.keys():
    if set_name not in legal_sets:
      continue
    if ignore is not None and set_name in ignore:
      continue
    if str(json_data[set_name]['type']) not in legal_set_types:
      continue
    for j in json_data[set_name]['cards']:
      if only_types is not None and 'types' in j and \
         all([str(t) not in only_types for t in j['types']]):
        continue
      try:
        c = convert_json_card(j, set_name, tokenizer)
        card_data.append(c)
      except Exception as e:
        error_cards += 1

  random.shuffle(card_data)
  card_data = { card[0] : card for card in card_data }
  if as_dictionary:
    return card_data

  card_data = card_data.values()
  if data_pct < 1:
    amt = int(len(card_data)*data_pct)
    card_data = card_data[:max(amt, 1)]

  test_amt = int(len(card_data)*test_pct)
  test_data = card_data[:test_amt]
  train_data = card_data[test_amt:]

  return train_data, test_data

def convert_json_card(json, set_name=None, tokenizer=None):
  name = json['name']
  cost = convert_cost(json.get('manaCost', ''))
  text = convert_text(json.get('text', ''), name)
  colors = convert_colors(json.get('colors', ''))
  card_types = convert_types(json.get('types', ''))
  if card_types[0]:
    power = convert_numeric(json.get('power'))
    toughness = convert_numeric(json.get('toughness'))
  else:
    power, toughness = 0,0
  if card_types[2]:
    loyalty = convert_numeric(json.get('loyalty'))
  else:
    loyalty = 0
  try:
    tokens = tokenizer.text_to_sequence(text)
  except Exception as e:
    tokens = []
  return Card(name, cost, text, card_types, power, toughness, loyalty, colors, set_name, tokens)

def convert_colors(colors):
  has_colors = np.zeros(6)
  for color in colors:
    if color == "White":
      has_colors[WHITE] = 1
    elif color == "Blue":
      has_colors[BLUE] = 1
    elif color == "Black":
      has_colors[BLACK] = 1
    elif color == "Red":
      has_colors[RED] = 1
    elif color == "Green":
      has_colors[GREEN] = 1
  return has_colors

def convert_numeric(val):
  return int(val)

def convert_types(card_types):
  ct_vec = np.zeros(8)
  for ct in card_types:
    ct = ct.lower()
    if ct == "creature":
      ct_vec[0] = 1
    elif ct == "enchantment" or ct == "enchant":
      ct_vec[1] = 1
    elif ct == "planeswalker":
      ct_vec[2] = 1
    elif ct == "sorcery":
      ct_vec[3] = 1
    elif ct == "instant":
      ct_vec[4] = 1
    elif ct == "artifact":
      ct_vec[5] = 1
    elif ct == "land":
      raise Exception("land sucks")
    elif ct == "tribal":
      ct_vec[7] = 1
    else:
      raise Exception("Bad card type")
  return ct_vec


def convert_text(text, name):
  x = "This is a sentence. (once a day)"
  text = re.sub("[\(.*?\)]", "", text)
  #Replace all occurances of name with %
  text = text.replace(name, '%')
  #Replace all newlines with a space
  text = text.replace('\n', ' ')
  #Convert to lower case
  text = text.lower().strip()
  
  return text

def convert_cost(text_cost):
  cost = np.zeros(2)
  for sym in mana_symbols(text_cost):
    if sym.isdigit() and cost[1] == 0:
      cost[1] += int(sym)
    else:
      cost[0] += 1
  return cost

  """
    oldcost function
  """
  cost = np.zeros(6)
  for sym in mana_symbols(text_cost):
    if sym == "W":
      cost[WHITE] += 1
    elif sym == "U":
      cost[BLUE] += 1
    elif sym == "B":
      cost[BLACK] += 1
    elif sym == "R":
      cost[RED] += 1
    elif sym == "G":
      cost[GREEN] += 1
    elif sym.isdigit() and cost[COLORLESS]==0:
      cost[COLORLESS] = int(sym)
    else:
      raise LookupError()
  return cost

def mana_symbols(cost):
  idx = 0
  while idx != -1:
    end = cost.find('}',idx)
    yield cost[idx+1:end]
    idx = cost.find('{',end)

def order_sets(data):
  from time import strptime
  set_list = []
  for set_name in data.keys():
    date_str = data[set_name]['releaseDate']
    date = strptime(date_str, '%Y-%m-%d')
    set_list.append((set_name, date))
  set_list.sort(key=lambda x: x[1])
  return map(lambda x: x[0], set_list)

def sets_before(set_order, set_name):
  idx = set_order.index(set_name)
  return set_order[:idx+1]

def sets_after(set_order, set_name):
  idx = set_order.index(set_name)
  return set_order[idx:]

def save_top_n(tokenizer, n=100):
  import operator
  sorted_x = sorted(tokenizer.word_counts.items(), key=operator.itemgetter(1), reverse=True)
  fd = open("top" + str(n) + ".txt", 'w')
  for word, num in sorted_x[:n]:
      print >> fd, word.encode('utf-8').strip()
  fd.close()

#!/usr/bin/env python2

from collections import namedtuple
import json
import random
import numpy as np

CARD_DATA = 'data/AllCards.json'
SET_DATA = 'data/AllSets.json'
WHITE, BLUE, BLACK, RED, GREEN, COLORLESS = range(6)

Card = namedtuple('Card', ['name', 'cost', 'text', 'types', 'power', 'toughness', 'loyalty', 'colors'])

"""
load_card_data
Generates a train and test data set.
Only loads `data_pct` percent of the total data set.
The test data set will be `test_pct` percent of all the data loaded.
Returns [training data], [testing data]

X - cost spells do not function. * cost power/toughness creatures do not function.
"""
def load_card_data(test_pct=0.1, data_pct=1, seed=1337):
  random.seed(seed)
  with open(CARD_DATA) as f:
    json_data = json.load(f).values()
  card_data = []
  error_cards = 0
  for j in json_data:
    try:
      c = convert_json_card(j)
      card_data.append(c)
    except Exception as e:
      error_cards += 1
  print len(card_data), error_cards
  random.shuffle(card_data)
  if data_pct < 1:
    amt = int(len(card_data)*data_pct)
    card_data = card_data[:max(amt, 1)]
  test_amt = int(len(card_data)*test_pct)
  test_data = card_data[:test_amt]
  train_data = card_data[test_amt:]
  return train_data, test_data

def convert_json_card(json):
  name = json['name']
  cost = convert_cost(json.get('manaCost', ''))
  text = convert_text(json.get('text', ''), name)
  colors = convert_colors(json.get('colors', ''))
  card_types = convert_types(json.get('types', ''))
  if card_types[0]:
    power = convert_numeric(json.get('power'))
    toughness = convert_numeric(json.get('power'))
  else:
    power, toughness = 0,0
  if card_types[2]:
    loyalty = convert_numeric(json.get('loyalty'))
  else:
    loyalty = 0
  return Card(name, cost, text, card_types, power, toughness, loyalty, colors)

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
      ct_vec[6] = 1
    elif ct == "tribal":
      ct_vec[7] = 1
    else:
      raise Exception("Bad card type")
  return ct_vec


def convert_text(text, name):
  #Replace all occurances of name with %
  text = text.replace(name, '%')
  #Replace all newlines with a space
  text = text.replace('\n', ' ')
  #Convert to lower case
  text = text.lower()
  return text

def convert_cost(text_cost):
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

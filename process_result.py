#!/usr/bin/env python2

import pickle
import numpy as np
from math import log

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PRED_FILE = 'output.p'
CMC_PENALTY = 5

def load_data(fname=PRED_FILE):
  with open(fname) as f:
    data = pickle.load(f)
  return data

def round_cost(cost):
  return list(map(round, cost))

def loss(pred, actual):
  np_pred, np_act = map(np.asarray, (pred, actual))
  first_log = np.log(np_pred+1)
  second_log = np.log(np_act+1)
  first_sum = np.log(np.sum(np_pred)+1)
  second_sum = np.log(np.sum(np_act)+1)
  return np.mean(np.square(first_log-second_log))+CMC_PENALTY*np.square(first_sum-second_sum)

def calc_losses(data):
  losses = []
  for d in data:
    r_pred = round_cost(d[0])
    act = d[1]
    l = loss(r_pred, act)
    losses.append(l)
  return losses

def to_cmc(pred):
  return int(sum(pred))

def group_by_cmc(data):
  cmc_groups = dict()
  for d in data:
    _, act = d
    cmc = to_cmc(act)
    if cmc not in cmc_groups:
      cmc_groups[cmc] = []
    cmc_groups[cmc].append(d)
  return cmc_groups

def avg_pred_cmc(data):
  cmcs = []
  for d in data:
    pred, _ = d
    cmcs.append(to_cmc(pred))
  return mean(cmcs)

def stddev_pred_cmc(data):
  cmcs = []
  for d in data:
    pred, _ = d
    cmcs.append(to_cmc(pred))
  cmcs = np.asarray(cmcs)
  return np.std(cmcs)

def mean(l):
  l = np.asarray(l)
  return np.mean(l).tolist()

def stddev(l):
  l = np.asarray(l)
  return np.std(l).tolist()

###

def loss_by_cmc(display=False):
  data = load_data()
  group_data = group_by_cmc(data)
  group_loss = dict()
  for cmc in group_data:
    losses = calc_losses(group_data[cmc])
    group_loss[cmc] = mean(losses)
  if display:
    for cmc in group_loss:
      print cmc, group_loss[cmc]
  return group_loss

def pred_by_cmc(display=False):
  data = load_data()
  group_data = group_by_cmc(data)
  group_avg = dict()
  for cmc in group_data:
    preds = group_data[cmc]
    group_avg[cmc] = (avg_pred_cmc(preds), stddev_pred_cmc(preds))
  if display:
    for cmc in group_data:
      print cmc, group_avg[cmc]
  return group_avg

###


def avg_graph():
  plt.clf()
  data = pred_by_cmc()

  ref = [0,10]

  cmcs = data.keys()
  avgs = [data[cmc][0] for cmc in cmcs]
  stds = [data[cmc][1] for cmc in cmcs]

  plt.errorbar(cmcs, avgs, yerr=stds, fmt='ro')
  plt.plot(ref, ref, 'b-')
  plt.savefig('avg_graph.png')

def loss_graph():
  plt.clf()
  data = loss_by_cmc()

  cmcs = data.keys()
  losses = [data[cmc] for cmc in cmcs]

  plt.plot(cmcs, losses, 'ro')
  plt.savefig('loss_graph.png')

###

if __name__=='__main__':
  #loss_by_cmc(True)
  #pred_by_cmc(True)
  avg_graph()
  loss_graph()

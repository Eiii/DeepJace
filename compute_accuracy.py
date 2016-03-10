#!/usr/bin/env python
import numpy as np
import sys
from scipy import mean, median, std

"""
Accepts a file which contains two vectors per line, first is predicted, second is true. We compute the 'accuracy' by my own human metrics, which is:

CMC = .5 point
n-colored = .25 point
Matching colors = .25 points

Obviously if you get them all correct, you get one point.

"""

scores = []
for line in open(sys.argv[1], 'r'):
    score = 0
    predicted, real = line.strip().split("\t")
    predicted = np.asarray(predicted.split(","), dtype='float32')
    real = np.asarray(real.split(","), dtype='float32')
    
    predicted = np.round(predicted)
    for i, val in enumerate(predicted):
        if val < 0:
            predicted[i] = 0
        
    "number of colored mana" 
    if sum(predicted[:-1]) == sum(real[:-1]):
        score += .32
    "correct colors"
    if max(real[:-1]) == 0:
        real_color_identity = real[:-1]
    else:
        real_color_identity = np.ceil(real[:-1]/max(real[:-1]))
    if max(predicted[:-1]) == 0:
        p_color_id = predicted[:-1]
    else:
        p_color_id = np.ceil(predicted[:-1]/max(predicted[:-1]))
    colors_correct = np.sum(abs(real_color_identity - p_color_id)) == 0
    
    if colors_correct:
        score += .33
    "correct cmc"
    if sum(predicted) == sum(real):
        score += .34
    scores.append(score)

print "accuracy(mean), median, std"
print mean(scores), median(scores), std(scores)
print scores.count(.99), len(scores)
print scores.count(.34), len(scores)
print scores.count(.33)
print scores.count(.32)
print len([something for something in scores if something > 0])
        

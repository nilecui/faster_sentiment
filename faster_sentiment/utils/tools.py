#########################################################################
#Author: nilecui
#Date: 2022-09-06 11:06:45
#LastEditors: nilecui
#LastEditTime: 2022-09-06 11:06:51
#FilePath: /faster_sentiment/utls/tools.py
#Description: 
#Details do not determine success or failure!
#Copyright (c) 2022 by nilecui, All Rights Reserved. 
#########################################################################

from __future__ import absolute_import, print_function, unicode_literals


def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def generate_set(x, number=1):
    res = []
    n_grams = set(zip(*[x[i:] for i in range(number)]))
    for n_gram in n_grams:
        res.append(n_gram)
    return res

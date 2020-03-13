import torch
import os
import argparse

import random

from generate import *

import pickle

# requires a GPU

if __name__ == '__main__':
    # list of models' names
    c = "comments.pt"
    g = "gaming.pt"
    m = "music.pt"
    n = "news.pt"
    
    fileNames = [c,g,m,n]
    
    #list of actual models
    decoder = [torch.load(c),
               torch.load(g),
               torch.load(m),
               torch.load(n)]
    
    # make list of corpus per model
    corpus = []
    for f in fileNames:
        file = open(f[0:len(f)-2] + "pkl", 'rb')
        temp = (pickle.load(file))
        file.close()
        corpus.append([(str(ele) + "\v") for ele in temp])
    
    for i in range(4):
        print("\nGenerating comment using " + fileNames[i] + ":\n")
        
        rand_val = int(random.random() * len(corpus[i]))
        
        print(generate(decoder[i], corpus[i], corpus[i][rand_val][0][0], 200, 0.8, True));
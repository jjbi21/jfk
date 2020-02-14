#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse

from helpers import *
from model import *

def generate(decoder, corpus, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    endOfComment = predicted.find("\v")
    endOfComment = endOfComment if endOfComment != -1 else len(predicted)
    predicted = predicted[0:endOfComment]

    for ele in corpus:
        if(predicted == ele):
            print("generated identical")
            predicted = generate(decoder, corpus, inp, predict_len, temperature, cuda)
            break

    return predicted
    
    #endOfComment = predicted.find("\v")
    #endOfComment = endOfComment if endOfComment != -1 else len(predicted)
    #return predicted[0:endOfComment]

# Run as standalone script
if __name__ == '__main__':

    
# Parse command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    args = argparser.parse_args()

    decoder = torch.load(args.filename)
    
    #construct corpus
    import pickle
    file = open(args.filename[0:len(args.filename)-2] + "pkl", 'rb')
    corpus = pickle.load(file)
    file.close()
    corpus = [(str(ele) + "\v") for ele in corpus]
    
    del args.filename
    #for i in range(500):
    print(generate(decoder, corpus, **vars(args)))
        #print("\n")


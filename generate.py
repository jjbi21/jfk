#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse

import random

from helpers import *
from model import *

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        softmax = nn.Softmax(dim=-1)
        cumulative_probs = torch.cumsum(softmax(sorted_logits), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

#added corpus as a way to handle /v
def generate(decoder, corpus, prime_str='A', predict_len=100, temperature=0.8, cuda=False, model='LSTM'):
    #init hidden layers
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    #if using cuda, set up relevant traits
    if cuda:
        hidden = hidden.cuda() if decoder.model == "gru" else (hidden[0].cuda(), hidden[1].cuda())
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    #get inputs
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        temperature = 1.0
        top_k = 0
        top_p = 0.9
        
        logits = output.data.view(-1).div(temperature)
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        # Sample from the filtered distribution
        softmax = nn.Softmax(dim=-1)
        probabilities = softmax(filtered_logits)
        next_token = torch.multinomial(probabilities, 1)

        # Add predicted character to string and use as next input
        predicted_char = all_characters[next_token]
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
    rand_val = int(random.random() * len(corpus))
    print(generate(decoder, corpus, corpus[rand_val][0][0], args.predict_len, args.temperature, args.cuda))
        #print("\n")

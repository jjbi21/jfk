#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch
# PURPOSE: Text generating script/function that loads in trained model and vector embeddings

import smart_open
import argparse

from helpers import *
from model import *


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    '''https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317'''
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


def generate(decoder, word_model, punc, prime_str='A', predict_len=100, temperature=0.8, cuda=False, model='lstm'):
    if type(predict_len)!=int:
        predict_len=int(predict_len)
    hidden = decoder.init_hidden(1)
    #prime_input = Variable(word_tensor([prime_str], word_model).unsqueeze(0))
    prime_input = Variable(word_tensor(prime_str, word_model).unsqueeze(0))

    if cuda:
        if model == 'lstm':
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        else:
            hidden = hidden.cuda()
        prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_input) ):
        _, hidden = decoder(prime_input[:,p], hidden)
        
    inp = prime_input[:,-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        # output_dist = output.data.view(-1).div(temperature).exp()
        # top_i = torch.multinomial(output_dist, 1)[0]

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
        #predicted_word = idx2word(top_i, word_model)
        predicted_word = idx2word(next_token, word_model)

        if punc:
            predicted += predicted_word
        else:
            predicted += ' ' + predicted_word
        inp = Variable(word_tensor([predicted_word], word_model).unsqueeze(0))
        if cuda:
            inp = inp.cuda()
        if punc:
            if '¬' in predicted:
                break
        else:
            if '▁' in predicted:
                break

    return predicted


def main(filename, predict_len, temperature, cuda, punc):
    decoder = torch.load(filename)
    if cuda:
        decoder.cuda()
    word_model = gensim.models.FastText.load(os.path.splitext(os.path.basename(filename))[0] + '.model')

    file = smart_open.open(os.path.splitext(os.path.basename(filename))[0] + '.sentences', 'rb')
    sentences = pickle.load(file)
    file.close()

    # script for testing longer priming strings
    # text_file = open("botchan.txt", "w")
    # text_file.write(' '.join([' '.join(sentence) for sentence in sentences]))
    # text_file.close()
    #
    # spm.SentencePieceTrainer.Train('--input=botchan.txt --model_prefix=m --hard_vocab_limit=false')
    # s = spm.SentencePieceProcessor()
    # s.Load('m.model')
    #
    # ss = []
    # for sentence in sentences:
    #     ss.append(s.encode_as_pieces(' '.join(sentence)))
    # sentences = ss
    # prime_str = s.encode_as_pieces('Tesla stock')
    # print(prime_str)
    # predicted = generate(decoder, word_model, punc, prime_str, predict_len, temperature, cuda)
    # print(''.join(predicted))

    while True:
        # loop breaks when a unique comment not found in training comments is generated
        rand_val = int(random.random() * len(sentences))
        prime_str = sentences[rand_val][0][0]
        predicted = generate(decoder, word_model, punc, prime_str, predict_len, temperature, cuda)
        if punc:
            for sentence in sentences:
                if ' '.join(predicted.split('▁')[1:]) in (' '.join(sentence)):
                    break
            else:
                break
        else:
            for sentence in sentences:
                if (' '.join(sentence)) == predicted:
                    break
            else:
                break
    if punc:
        print(' '.join(predicted.split('▁')[1:]))
    else:
        print(predicted)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--punc', action='store_true')
    args = argparser.parse_args()
    main(args.filename, args.predict_len, args.temperature, args.cuda, args.punc)
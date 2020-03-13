# https://github.com/spro/char-rnn.pytorch
# PURPOSE: helper functions, mainly to perform necessary preprocessing and train vector embeddings

import random
import time
import math
import torch
import pickle
import collections
import gensim
import os
import sentencepiece as spm
from nltk.tokenize import TweetTokenizer

# Reading and un-unicode-encoding data

max_sentence_len = 40

def read_file(filename, p):
    '''performs tokenization with sentencepiece or NLTK and then generates word model using FastText'''
    if os.path.exists(os.path.splitext(os.path.basename(filename))[0] + ".sentences"):
        text_file = open(os.path.splitext(os.path.basename(filename))[0] + ".sentences", 'rb')
        sentences = pickle.load(text_file)
        text_file.close()
    else:
        file = open(filename, 'rb')
        data = pickle.load(file)
        file.close()
        if p:
            comments = [item + ' ¬\n' for item in data]
        else:
            comments = [item + ' ▁' for item in data]
        if p:
            comments = random.choices(comments, k=40000)
            sentences = [[word for word in doc.split(' ')[:max_sentence_len] if word != ''] for doc in comments]
        else:
            comments = random.choices(comments, k=40000)
            tk = TweetTokenizer()
            for i in range(len(comments)):
                comments[i] = tk.tokenize(comments[i])
            sentences = comments
        sentences = list(filter(None, sentences))
        text_file = open(os.path.splitext(os.path.basename(filename))[0] + ".sentences",'wb')
        pickle.dump(sentences, text_file)
        text_file.close()

    if p:
        text_file = open("botchan.txt", "w")
        text_file.write(' '.join([' '.join(sentence) for sentence in sentences]))
        text_file.close()

        spm.SentencePieceTrainer.Train('--input=botchan.txt --model_prefix=m --hard_vocab_limit=false')
        s = spm.SentencePieceProcessor()
        s.Load('m.model')

        ss = []
        for sentence in sentences:
            ss.append(s.encode_as_pieces(' '.join(sentence)))
        sentences = ss

    freq = collections.defaultdict(int)
    for sentence in sentences:
        if len(sentence):
            freq[sentence[0]] += 1
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
    priming_str = list(zip(*(sorted_freq[:3])))[0]

    if os.path.exists(os.path.splitext(os.path.basename(filename))[0] + ".model"):
        word_model = gensim.models.FastText.load(os.path.splitext(os.path.basename(filename))[0] + ".model")
    else:
        if p:
            word_model = gensim.models.FastText(sentences, size=300, min_count=1, window=7, iter=200, sg=1)
        else:
            word_model = gensim.models.FastText(sentences, size=300, min_count=1, window=5, iter=100, sg=0)

    file = [word for sentence in sentences for word in sentence]
    file_len = sum(freq.values())
    pretrained_weights = word_model.wv.vectors
    vocab_size, embedding_size = pretrained_weights.shape
    word_model.save(os.path.splitext(os.path.basename(filename))[0] + ".model")
    return file, file_len, pretrained_weights, word_model, vocab_size, embedding_size, priming_str

# Turning a string into a tensor
def word_tensor(words, word_model):
    tensor = torch.zeros(len(words)).long()
    for i in range(len(words)):
        tensor[i] = word_model.wv.vocab[words[i]].index

    return tensor
def word2idx(word, word_model):
    return word_model.wv.vocab[word].index
def idx2word(idx, word_model):
    return word_model.wv.index2word[idx]



# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


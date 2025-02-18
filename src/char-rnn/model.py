# https://github.com/spro/char-rnn.pytorch

# modified slighty for bidirectional implementation
# Removed code for gru for clarity

import torch
import torch.nn as nn
from torch.autograd import Variable

class CharRNN(nn.Module): #removed gru option from original source code
    def __init__(self, input_size, hidden_size, output_size, model="lstm", n_layers=1, bidirectional = True):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, bidirectional=True)
        self.decoder = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers * 2, batch_size, self.hidden_size)))

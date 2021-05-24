"""
ENCODER:
- Single GRU, but we will use bidirectional RNN
- Bidirectional RNN = have 2 RNNs in each layers.
    + forward RNN goes over the embedded sentence from left to right
    + backward RNN goes over the embbedd sentence from right to left
    + bidirectional
- We pass input embedded to RNN which tell Pytorch to initialize both the fw and bw initial hidden state to a tensor of all zeros.
- We will get 2 context vector, 1 for forward RNN after it has seen the final word in a sentence, and one from the backward rnn after it has seen the frist word in the senten


""" 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        # sropout = dropout rate
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        self.fc = nn.Linear(enc_hid_dim*2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout) # during training, randomly zeroes some of the elements of the input tensor with proability p using samples from Bernoulli distribution. This has proven to be an effective technique for regularization and preventing the coadaptation of neurons

    def forward(self, src):
        #src = [src len, batch size]
        embedded = self.dropout(self.embedding(src))         #embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)

        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 

        # encoder ENNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]

        return outputs, hidden

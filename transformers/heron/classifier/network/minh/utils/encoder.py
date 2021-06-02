import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

from collections import OrderedDict
from typing import Dict
from typing import Tuple

# Due to testing the gated architecture, there is no need to have it compatible with Adept's ModularNetwork
# Note that our architecture right now only have 1 layer of encoder and decoder and single head encoder/decoder
# Might want to have multiple encoder and decoder
# Might want to consider multi head encoder and decoder
# Read the paper and see what architecture does the Transformer works

# Resource to consider from the paper: https://github.com/dhruvramani/Transformers-RL/blob/master/layers.py
# Resource to consider from the paper: https://github.com/jerrodparker20/adaptive-transformers-in-rl/blob/master/StableTransformersReplication/transformer_xl.py


# Class Encoder acts as a wrapper that allow to be reuse EncoderLayers multiple times
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_heads, pf_dim, dropout, device, n_layers = 1, max_length = 100):
        """Acts as a wrapper that process the input text data, allows to reuse the EncoderLayer multiple times, and provides output for the Decoder
        
        Parameters
        ----------
        input_dim: 
            Tokenized input sentence to be passed into the networl
        hid_dim: 
            Processed input to be passed into the EncoderLayer
        n_head:
            Number(s) of head for attention mechanism, can be single or multihead
        pf_dim:
            Input dimension of the Projection Layer, this is usually larger than the hid_dim
        dropout:
            During training, randomly zeroes some elements of the input tensor with probability p=0.5 using samples from the Bernoulli distribution
        device:
            CPU or GPU
        n_layers:
            Number(s) of EncoderLayer layers we should use
        max_length:
            The positional embedding is initialized to have a vocab of 100. This can be increased if used on a dataset with longer sequence
        """
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim) #(input, output)
        self.pos_embedding = nn.Embedding(max_length, hid_dim) #(input, output)
        
        # This design is for multilayer encoder
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, pf_dim, dropout, device, n_heads) for _ in range(n_layers)])        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device=device)
      
    def forward(self, src, src_mask):
        """The forward function for encoding layer

        Parameters
        ----------
        src:
            Processed (tokenized & vectorized) text
        src_mask:
            Source mask is simply the same shape as the source sentence but has the value of 1 when the toen in the source sentence
                is not <pad> token and 0 when it is a <pad> token. 
            This is used in the encoder layers to mask the multi-head attention mechanism, which are used to calculate and apply attentio over 
                the source sentence, so the model does not pay attention to <pad> tokens, which contain no useful information
        
        Returns
        ----------
        src:
            Output of the encoder with the shape of [batch_size, src_len, hid_dim]
        """
        # src = [batch_size, src_len]
        # src_mask = [batch_size, 1,1, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional vector 
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch_size, src_len]

        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src = [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch_size, src_len, hid_dim]

        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout, device, n_heads = 1):
        """Acts as a wrapper that contain the LayerNorm, Self-Attention, Gating Layer, Projection/Feedforward Layer
        The design allow to stack up multiple EncoderLayer

        Parameter
        ----------
        hid_dim:
            Dimension of the processed (tokenized & vectorized) text data that will be input to the LayerNorm
        pf_dim:
            Dimension of the feedforward Projection Layer
        dropout:
            During training, randomly zeroes some elements of the input tensor with probability p=0.5 using samples from the Bernoulli distribution
        device:
            Can be CPU or GPU
        """
        super().__init__()
        self.first_norm= LNorm(hid_dim)
        self.attn_layer = Attn(hid_dim, dropout, device)
        self.first_gate = Gate(hid_dim)
        self.second_norm = LNorm(hid_dim)
        self.projection = Projection(hid_dim, pf_dim, dropout)
        self.second_gate = Gate(hid_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """The forward function for EncoderLayer

        Parameter
        ----------
        src:
            Processed (tokenized & vectorized) text
        src_mask:
            Source mask is simply the same shape as the source sentence but has the value of 1 when the toen in the source sentence
                is not <pad> token and 0 when it is a <pad> token. 
            This is used in the encoder layers to mask the multi-head attention mechanism, which are used to calculate and apply attentio over 
                the source sentence, so the model does not pay attention to <pad> tokens, which contain no useful information
        
        Returns
        ----------
        second_gate_output:
            Same dimension as the src input [batch size, src len, hid dim]
        """

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 

        first_norm_out = self.first_norm(src)
        attn_out, _ = self.attn_layer(first_norm_out, first_norm_out, first_norm_out, src_mask)
        first_gate_out = self.first_gate(src, self.dropout(attn_out))

        second_norm_out = self.second_norm(first_gate_out)
        projection_out = self.projection(second_norm_out)
        second_gate_out = self.second_gate(projection_out, self.dropout(first_gate_out))

        # second_gate_out = [batch size, src len, hid dim]
        return second_gate_out

class LNorm(nn.Module):
    def __init__(self, input_dim):
        """Class LayerNorm that applies layer normalization on the input. It normalized the activations of the layer 
            for each given input in the batch independently, rather than acroos a batch like Batch Norm

        Parameter
        ----------
        input_dim:
            [batch size, src len, hid dim]
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """The forward function for the LNorm Layer

        Parameter
        ----------
        x:
            Input to be passed in the model
        
        Return
        ----------
        x:
            Normalized input
        """
        x = self.layer_norm(x)
        return x    

class Attn(nn.Module):
    def __init__(self, hid_dim, dropout, device, scale = True, n_heads=1):
        """Multi-head self attention layer for transformer

        Parameter
        ----------
        hid_dim:
            Dimension of the processed (normalized, tokenized & vectorized) text data that will be input to the LayerNorm
        dropout:
            During training, randomly zeroes some elements of the input tensor with probability p=0.5 using samples from the Bernoulli distribution
        device:
            Can be CPU or GPU
        scale: bool
            Whether or not to inclue constant C (or d_k) in the equation sigmoid(q*k^T / C)
        dropout:
            Probability of dropout
        """
        super().__init__()

        assert hid_dim % n_heads == 0 # make sure that the multi-head dim is dividable

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device) # d_k
        # self.scale = self.hid_dim ** 0.5 if scale else 1.0 # TODO: Try this later if ^ works

    def forward(self, query, key, value, mask = None):
        """The forward function of the Attention Layer

        Parameter
        ----------
        query, key, value:
            Query is used with the key to get the attention vector which is then used to get the weighted sum of the values
        mask: bool
            We then mask the energy so we do not pay attention over any elements of the sequeuence we shouldn't

        Returns
        ----------
        x: output for the Gate Layer
            [batch size, query len, hid dim]
        """
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        # Change the shape of Q,K,V vector
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1) # attention = softmax of QK/d_k
        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)
        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous() # Change the order in the shape, make sure that the rearrange array still contiguous 
        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim) # combine the heads together
        #x = [batch size, query len, hid dim]

        x = self.fc_o(x) # linear output layer for attention
        #x = [batch size, query len, hid dim]

        return x, attention # we don't need attention for our case

# TODO: Might have to change this to ReLU
class Projection(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        """Feedforward Positionwise Multilayer Perceptron for the EncoderLayer

        Parameter
        ----------
        hid_dim:
            Dimension of the processed (gated, normalized, tokenized & vectorized) text data that will be input to the LayerNorm
        pf_dim:
            Input dimension of the Projection Layer, this is usually larger than the hid_dim
        dropout:
            Probability of dropout
        """
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """The forward function for Projection/Positionwise MLP Layer

        Parameter
        ----------
        x: 
            output of the LayerNorm 
        
        Return
        ----------
        x:
            Output of the Projection Layer that has the dim of [batch size, seq len, hid dim]. This will be an input for the Gating Layer
        """
        # x = [batch size, seq len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x)))
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        #x = [batch size, seq len, hid dim]

        return x

class Gate(nn.Module): # make sure this has the input dim and the self attention dim to be the same size
    def __init__ (self, input_dim):
        """Gating Layer for the EncoderLayer
    
        Parameter
        ----------
        input_dim:
            Dimension of the processed (feedforwarded, gated, normalized, tokenized & vectorized) text data that will be input to the LayerNorm

        Return
        ----------
        gate_output:
            output of the second Gating Layer that has a dim of [batch size, seq len, hid dim]
        """
        super().__init__()
        self.gru = nn.GRU(input_size = input_dim, hidden_size = input_dim) # output the hidden state & gate output

    def forward(self, x, y): # x is the original input, y is the attention output
        """The feedforward functio for the Gating Layer

        Parameter
        ----------
        x: 
            The output from the first gating layer
        Return
        ----------
        gate_output:
            output of the second Gating Layer that has a dim of [batch size, seq len, hid dim]
        """
        gate_output, _ = self.gru(x, y)
        #gate_output = [batch size, seq len, hid dim]

        return gate_output

###############################################################################
def encoder():
    print("Runnning Encoder")

if __name__ == "__main__":
    encoder()




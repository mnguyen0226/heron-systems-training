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

"""
About: The mode is made up of an encoder and decoder with encoder encode input/source sentence into context vector
    and decoder then decode the context vector to output our output/target sentence

ENCODER:
- Does not attempt to compress the entire source sentence into a singlle context vector, instead, it products a sequence of context vectors.
    Why this is not called a sequence of hidden state? 
        + A hidden state at time t in RNN has only seen tokens x_t and all the tokens before it
        + However, each context vector here has seen all tokens at all positions within the input sequence
- How it works?
    + First, the tokens are passed thru a standard embedding layer     
    + Next, as the model has no recurrent, it has no idea about the order of the tokens within the sequence.
        We solve this by using a second embedding layer called a positional embedding layer
        The position embedding has a vocab size of 100 meaning that the model can accept sentences up to 100 tokens long
        This 100 can increase if we want to handle longer sentence
    + The token and positional embedding are elementwise summed together to get a vector which contains info about the tokens and also its position with in the sequence
    + the combined embeddings are then passed thru N encoder layers to get Z, which is then output and can be used by decoder 
"""


"""
ENCODER LAYER:
- We first pass the src sentence and its mask into the multi-head attention layer, then perform a dropout on it, apply a sedifual connection and pass it thru a Layer Normalization  layer
- We then pass it thru a position-wise feedforward layer and then again apply dropout, a residual connection and then a layer normalization to get the output of this layer which is fed into the next layer.
- The muti-head attention layer is used by encoder layer to attend to the source sentence meaning that it calculate and applies attention over itself instead of another sequence, hence we call it self-attention
- This basically jsut resolve input and call the encoder sublayers but not implement the sublayer yet. It does allow for repetition tho
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()
        self.device = device 
        self.tok_embedding = nn.Embedding(input_dim, hid_dim) # input, output
        self.pos_embedding = nn.Embedding(max_length, hid_dim) # input, output

        # this is submodule that can be repeat 6 times
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
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

"""
ENCODER LAYER - ENCODER SUBLAYERS
- We first pass the srouce sentence and its mask into the multi-head attention layer, the perform drop out on it.
- We the apply the residual connection and pass thru the layer normalization layer.
- We then pass it thru a position-wise feedforward laer and then apply dropout, a residual connection and a normalization to get the output of this SUBLAYER which will then feed to a next sublayer.

The multihead attention layer is used by the encoder layer to attend to the srouce sentece = it calculating and applying attention over itself intead of another sequence =? self attention
"""
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 

        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]

        return src




def model():
    print("Runnning Model")

if __name__ == "__main__":
    model()
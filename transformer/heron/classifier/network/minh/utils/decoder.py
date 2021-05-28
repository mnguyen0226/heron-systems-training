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

from utils.encoder import Attn
from utils.encoder import Encoder
from utils.encoder import EncoderLayer
from utils.encoder import LNorm
from utils.encoder import Gate
from utils.encoder import Projection

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_heads, pf_dim, dropout, device, n_layers = 1, max_length = 100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim, pf_dim, dropout, device, n_heads) for _ in range (n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device=device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)           
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))    
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        #output = [batch size, trg len, output dim]
            
        return output, attention # we won't need attention for our case

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        # Layers handle the auto-regressive encoding of the Decoder
        self.auto_norm = LNorm(hid_dim)
        self.auto_attn = Attn(hid_dim, dropout, device)
        self.auto_gate = Gate(hid_dim)

        # This section is almost identical to the encoder, but instead of using the output of the 
        # previous layers for q,k,v; we instead use the encoder's k & v and use q from the decoder
        self.second_norm = LNorm(hid_dim)
        self.self_attention = Attn(hid_dim, dropout, device)
        self.second_gate = Gate(hid_dim)

        self.third_norm = LNorm(hid_dim)
        self.projection = Projection(hid_dim, pf_dim, dropout)
        self.third_gate = Gate(hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        first_norm_out = self.auto_norm(trg)
        masked_attention, _ = self.auto_attn(first_norm_out, first_norm_out, first_norm_out, trg_mask)
        first_gate_out = self.auto_gate(trg, self.dropout(masked_attention))

        second_norm_out = self.second_norm(first_gate_out)
        self_attention, attention = self.self_attention(second_norm_out, enc_src, enc_src, src_mask)
        second_gate_out = self.second_gate(first_gate_out, self.dropout(self_attention))

        third_norm_out = self.third_norm(second_gate_out)
        projection_out = self.projection(third_norm_out)
        third_gate_out = self.third_gate(second_gate_out, self.dropout(projection_out))

        return third_gate_out, attention

###############################################################################
def decoder():
    print("Runnning decoder")

if __name__ == "__main__":
    decoder()



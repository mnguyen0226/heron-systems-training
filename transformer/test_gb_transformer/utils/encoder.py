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

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        """Encoder class for Gated Transformer which is used for
            + embedding preprocessed input texts
            + calling n_layers EncoderLayers
            + providing encoded output for Decoder

        Parameters
        ----------
        input_dim:
            input dimension of the tokenized text to Input Embedding layer
        hid_dim:
            dimension of the output of Input Embedding layer and input to EncoderLayer layer
        n_layers:
            number of layer(s) of the EncoderLayer
        pf_dim:
            dimension of the output from the Feedforward layer
        dropout:
            dropout rate = 0.1
        device: 
            cpu or gpu
        max_length:
            the Input Embedding's position embedding has a vocab size of 100 which means our model can accept sentences up to 100 tokens long 
        """
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=hid_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=max_length, embedding_dim=hid_dim)

        self.layers = nn.ModuleList([GatedEncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device) # sqrt(d_model)

    def forward(self, src, src_mask):
        """Forwards function for the Encoder

        Parameters
        ----------
        src: [batch_size, src_len]
            tokenized vector input text
        src_mask: [batch_size, 1, 1, src_len]
            masked the input text but allow to ignore <pad> after tokenized during training in the tokenized vector since it does not provide any value

        Return 
        ----------
        src: [batch_size, src_len, hid_dim]. This is the dimension that will be maintain till output of Decoder
            position-encoded & embedded output of the encoder layer. The src will be fetched into the Decoder
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional vector. pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # src = [batch_size, src_len, hid_dim]. Here we dropout the input source so we have to dropout before doing Gating Layer
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)) 

        for layer in self.layers:
            # src = [batch_size, src_len, hid_dim]
            src = layer(src, src_mask)

        return src

class GatedEncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        """Gated Encoder layer of Encoder of the Transformer

        Parameters
        ----------
        hid_dim:
            input hidden_dim from the processed positioned-encoded & embedded vectorized input text
        n_heads:
            number of head(s) for attention mechanism
        pf_dim:
            input feed-forward dimension
        dropout:
            dropout rate = 0.1
        device:
            cpu or gpu
        """
        super().__init__()
        self.first_layer_norm = nn.LayerNorm(normalized_shape=hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.first_gate = nn.GRU(input_size=hid_dim, hidden_size=hid_dim)
        self.second_layer_norm = nn.LayerNorm(normalized_shape=hid_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.second_gate = nn.GRU(input_size=hid_dim, hidden_size=hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        """Feed-forward layer for the Gate Encoder layer
        
        Parameters
        ----------
        src: [batch size, src len, hid dim]
            tokenized vector input text
        src_mask: [batch_size, 1, 1, src_len]
            masked the input text but allow to ignore <pad> after tokenized during training in the tokenized vector since it does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]. This is the dimension that will be maintain till output of Decoder
            position-encoded & embedded output of the encoder layer. The src will be fetched into the Decoder       
        """
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 

        # first layer norm - already dropped out from Encoder class
        src = self.first_layer_norm(src)

        # self-attention
        _src, _ = self.self_attention(query=src, key=src, value=src, mask=src_mask)

        # first gate
        first_gate_output, _ = self.first_gate(self.dropout(_src), src) # [batch size, src len, hid dim]

        # second layer norm - already dropped from first gate
        src = self.second_layer_norm(first_gate_output) 

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # second gate
        second_gate_output, _ = self.second_gate(self.dropout(_src), src) # [batch size, src len, hid dim]

        return second_gate_output

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        """Multi/single Head Attention Layer. This layer define Q,K,V of the GateEncoderLayer

        Parameters
        ----------
        hid_dim:
            input hidden dimension from the first layer norm
        n_heads:
            number of heads for attention mechanism
        dropout:
            dropout rate = 0.1
        device: 
            cpu or gpu
        """

        super().__init__()

        assert hid_dim % n_heads == 0 # make sure that number of multiheads are concatenatable

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads # determine the head_dim

        self.fc_q = nn.Linear(in_features=hid_dim, out_features=hid_dim) # apply linear transformation
        self.fc_k = nn.Linear(in_features=hid_dim, out_features=hid_dim) # apply linear transformation
        self.fc_v = nn.Linear(in_features=hid_dim, out_features=hid_dim) # apply linear transformation

        self.fc_o = nn.Linear(in_features=hid_dim, out_features=hid_dim) # apply linear transformation

        self.dropout = nn.Dropout(dropout) 
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device) # d_k = head_dim, will be hid_dim if n_heads == 1

    def forward(self, query, key, value, mask=None):
        """Feed-forward layer for the attention mechanism

        Parameters
        ----------
        query, key, value:
            Query is used with Key to get an attention vector which is then weighted sum with Value
        mask:
            masked the input text but allow to ignore <pad> after tokenized during training in the tokenized vector since it does not provide any value

        Return
        ----------
        x: [batch size, query len, hid dim] 
            input to the first gate layer
        """
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
        
        batch_size = query.shape[0]

        Q = self.fc_q(query) # applied linear transformation but keep dim, Q = [batch size, query len, hid dim]
        K = self.fc_k(key) # applied linear transformation but keep dim, K = [batch size, key len, hid dim]
        V = self.fc_v(value) # applied linear transformation but keep dim, V = [batch size, value len, hid dim]

        # Change the shape of QKV
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) #Q = [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) #K = [batch size, n heads, key len, head dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) #V = [batch size, n heads, value len, head dim]

        #-------------------------------------------------------
        #energy = [batch size, n heads, query len, key len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale # matmul, scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10) # mask
        
        #attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim = -1) # attention = softmax of QK/d_k
        
        #x = [batch size, n heads, query len, head dim] # original Q,K,V dim
        x = torch.matmul(self.dropout(attention), V) # matmul
        #-------------------------------------------------------

        #x = [batch size, query len, n heads, head dim]
        x = x.permute(0, 2, 1, 3).contiguous() # Change the shape again for concat
        
        #x = [batch size, query len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim) # combine the heads together

        #x = [batch size, query len, hid dim]
        x = self.fc_o(x) # Linear layer output for attention
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        """Positionwise Feedforward layer of GatedEncoderLayer
        Why is this used? Unfortunately, it is never explained in the paper.
        The transformed from hid_dim to pf_dim (pf_dim >> hid_dim.
        The ReLU activation function and dropout are applied before it is transformed back into hid_dim representation
        
        Parameters
        ----------
        hid_dim:
            input hidden dimension from the second layer norm
        pf_dim:
            dimension of the output for the position-wise feedforward layer
        dropout:
            dropout rate = 0.1
        """
        super().__init__()
        self.fc_1 = nn.Linear(in_features=hid_dim, out_features=pf_dim) # linear transformation
        self.fc_2 = nn.Linear(in_features=pf_dim, out_features=hid_dim) # linear transformation # make sure to conert back from pf_dim to hid_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Feedforward function for the PFF layer

        Parameters
        ----------
        x: [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
            input from the second layer norm
        
        Return
        ---------- 
        x: [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
            output to the second gate layer
        """
        #x = [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
        x = self.dropout(torch.relu(self.fc_1(x))) # relu then dropout to contain same infor

        #x = [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
        x = self.fc_2(x)

        return x


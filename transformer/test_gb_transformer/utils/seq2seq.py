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

from utils.preprocess import *
from utils.encoder import Encoder
from utils.decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        """Seq2Seq encapsulates the encoder and decoder and handle the creation of masks (for src and trg)

        Parameters
        ----------
        encoder:
            the Encoder layer
        decoder:
            the Decoder layer
        src_pad_idx:

        trg_pad_idx:

        device:
            cpu or gpu
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        """Making input source mask by checking where the source sequence is not equal to a <pad> token
            It is 1 where the token is not a <pad> token and 0 when it is
        
        Parameters
        ----------
        src: [batch size, src len]
            input training tokenized source sentence(s)

        Return
        ----------
        src_mask: [batch size, 1, 1, src len]
            mask of the input source
        """
        #src = [batch size, src len]
    
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        """Making a target mask similar to srouce mask. Then we create a subsequence mask trg_sub_mask.
            This creates a diagonal matrix where the elements above the diagonal will be 0 and the elements below the diagonal will be set to
            whatever the input tensor is.

        Parameters
        ----------
        trg: [batch size, trg len]
            target tokens/labels

        Return
        ----------
        trg_mask: [batch size, 1, trg len, trg len]   
            mask of the target label
        """
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        """Feed-forward function of the Seq2Seq

        Parameters
        ----------
        src: [batch size, src len]
            input source (to Encoder)
        trg: [batch size, trg len]
            output label (from Decoder)

        Return
        ----------
        output: [batch size, trg len, output dim]
            output prediction
        attention: [batch size, n heads, trg len, src len]
            we will not care about this in our case
        """
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
###############################################################################
def test_seq2seq():
    print("Runnning Seq2Seq")

if __name__ == "__main__":
    test_seq2seq()
# We have to write extra function for to wrap

# For Encoder
# Create a class Embedding:
    # Tokenized and numerical Embedding layer

# Dropout and scale will be used in the Seq2Seq class

#######################################################
# For Decoder 
# Create a class Embedding:
    # Tokenized and numerical Embedding layer   

# Dropout and scale will be used in the Seq2Seq class

import typing import Tuple
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int):
        """ Tokenize and Positional Encodiing

        input_dim: int
            input dimension of the tokenized text to input embedding layer
        hid_dim: int
            dimension of the output of input Embedding layer and input to the Encoder layer

        Return
        ----------
        src: [batch_size, src_len]
            input the
        """
        super().__init__()
        self.tok_embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=hid_dim)
        self.pos_embedding = nn.Embedding(num_embeddings=100, embedding_dim=hid_dim)

    def forward(self, src, src_mask):
        """Forward function for the Embedding Layer 
        """
        batch_size = src.shape[0] # this maybe different
        src_len = src.shape[1] # this  maybe different

        # positional vector, pos = [batch_size, src_len]
        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1) # device
        )

        # src = [batch_size, src_len, hid_dim]. Here we dropout the input source so we have to dropout again before the Gating layer
        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )

        return src

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder ,src_pad_idx, trg_pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder 
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

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
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]
        return src_mask

    def make_trg_mask(self, trg):
        """Making a target mask similar to srouce mask. Then we create a subsequence mask trg_sub_mask.
            This creates a diagonal matrix where the elements above the diagonal will be 0 and the elements
                below the diagonal will be set to
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
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()
        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src):
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
        src_mask = self.make_src_mask(src=src)
        trg_mask = self.make_trg_mask(trg=trg)
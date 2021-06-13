# Here is where we have to preprocess the data, get the right shape to Embedding, Encoder, Decoder as well as any additional layers for the output

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

from typing import Tuple
import torch
import torch.nn as nn
from utils.preprocess import device


class Embedding(nn.Module):
    # this just encode the batch size, features

    def __init__(
        self, input_dim: int, hid_dim: int, dropout: float
    ):  # these two are features
        super().__init__()
        self.tok_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hid_dim
        )
        self.pos_embedding = nn.Embedding(num_embeddings=100, embedding_dim=hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = hid_dim ** 0.5

    def forward(self, src):  # return the embedding source before the encoder layer
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional vector. pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # src = [batch_size, src_len, hid_dim].
        emb_src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )

        return emb_src


class Seq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder, src_pad_idx, trg_pad_idx):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        emb_src = self.embedding(src)  # src = [batch_size, src_len],

        # emb_src = [batch_size, sequence, features] = [batch, # word in a sentence, features = number of float]
        print("TESTING Seq2Seq forward function")
        print(src.shape)
        print(emb_src.shape)

        emb_src = emb_src.permute(0, 2, 1)
        print(
            emb_src.shape
        )  # we want the shape to be similar to the one in the Encoder's forward function

        # this will be changed. Some how could encode the src_mask and the trg_mask
        enc_src = self.encoder(emb_src, src_mask)

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention

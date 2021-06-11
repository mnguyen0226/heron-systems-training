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

    def forward(self, src):
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional vector. pos = [batch_size, src_len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        # src = [batch_size, src_len, hid_dim].
        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )

        return src


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

        print(f"TESTING: SRC Shape {src.shape}")
        emb_src = self.embedding(src)

        print(f"TESTING: EMB Shape {emb_src.shape}")

        b, s, f = emb_src.shape

        emb_src_reshape = torch.reshape(
            emb_src.permute(0, 2, 1), (f, b * s)
        )  # batch = batch. sequence = number of word in a sentence, features = number of float representing a word

        print(f"TESTING: EMB RESHAPE Shape {emb_src_reshape.shape}\n")
        # use Encoder here

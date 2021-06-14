# script run the Seq2Seq of the Gated Transformers

from typing import Tuple
import torch
import torch.nn as nn
from utils.preprocess import device


class EmbeddingEncLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout):
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

        # positional vector
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )

        return src


class EmbeddingDecLayer(nn.Module):
    def __init__(self, output_dim, hid_dim, dropout):
        super().__init__()
        self.tok_embedding = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=hid_dim
        )
        self.pos_embedding = nn.Embedding(num_embeddings=100, embedding_dim=hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = hid_dim ** 0.5

    def forward(self, trg):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # positional vector
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )

        return trg


class GatedSeq2Seq(nn.Module):
    def __init__(
        self,
        embedding_enc: Tuple[int, int, float],
        embedding_dec: Tuple[int, int, float],
        encoder: Tuple[int, int, int, int, int, float, str],
        decoder: Tuple[int, int, int, int, int, float, str],
        src_pad_idx: Tuple[list, str, str, bool, bool],
        trg_pad_idx: Tuple[list, str, str, bool, bool],
    ):
        """Seq2Seq encapsulates the encoder and decoder and handle the creation of masks (for src and trg)

        Parameters
        ----------
        encoder: [input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length]
            the Encoder layer
        decoder: [output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, max_length]
            the Decoder layer
        src_pad_idx:
            type Field (preprocess.py)
        trg_pad_idx:
            type Field (preprocess.py)
        """
        super().__init__()
        self.embedding_enc = embedding_enc
        self.embedding_dec = embedding_dec
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

    def forward(
        self, src: Tuple[int, int], trg: Tuple[int, int]
    ) -> Tuple[tuple, tuple]:
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
        # src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        embedding_src_enc = self.embedding_enc(src=src)  # B S F

        # print("SEQ2SEQ: Finish Embedding Encoder----------------------------- \n")

        embedding_src_enc = embedding_src_enc.permute(0, 2, 1)  # B F S
        # print(f"SEQ2SEQ: {embedding_src_enc.shape}")

        # enc_src = self.encoder(embedding_src_enc, src_mask)
        enc_src = self.encoder(embedding_src_enc)
        # enc_src = [batch size, src len, hid dim]

        # print("SEQ2SEQ: Finish Encoding----------------------------- \n")

        embedding_trg_dec = self.embedding_dec(trg=trg)
        embedding_trg_dec = embedding_trg_dec.permute(0, 2, 1)  # B F S

        # print(f"SEQ2SEQ: {embedding_trg_dec.shape}")
        # print("SEQ2SEQ: Finish Embedding Decoder----------------------------- \n")

        # output, attention = self.decoder(embedding_trg_dec, enc_src)
        output = self.decoder(embedding_trg_dec, enc_src)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        # print(f"SEQ2SEQ output: {output.shape}")

        # print("SEQ2SEQ: Finish Decoding----------------------------- \n")

        # return output, attention
        return output, 1

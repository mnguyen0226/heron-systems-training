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


class Seq2Seq(nn.Module):
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

    # def make_src_mask(self, src: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    #     """Making input source mask by checking where the source sequence is not equal to a <pad> token
    #         It is 1 where the token is not a <pad> token and 0 when it is

    #     Parameters
    #     ----------
    #     src: [batch size, src len]
    #         input training tokenized source sentence(s)

    #     Return
    #     ----------
    #     src_mask: [batch size, 1, 1, src len]
    #         mask of the input source
    #     """
    #     # src = [batch size, src len]

    #     src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
    #     # src_mask = [batch size, 1, 1, src len]

    #     return src_mask

    # def make_trg_mask(self, trg: Tuple[int, int]) -> Tuple[int, int, int, int]:
    #     """Making a target mask similar to srouce mask. Then we create a subsequence mask trg_sub_mask.
    #         This creates a diagonal matrix where the elements above the diagonal will be 0 and the elements
    #             below the diagonal will be set to
    #         whatever the input tensor is.

    #     Parameters
    #     ----------
    #     trg: [batch size, trg len]
    #         target tokens/labels

    #     Return
    #     ----------
    #     trg_mask: [batch size, 1, trg len, trg len]
    #         mask of the target label
    #     """
    #     # trg = [batch size, trg len]

    #     trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
    #     # trg_pad_mask = [batch size, 1, 1, trg len]

    #     trg_len = trg.shape[1]

    #     trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).bool()
    #     # trg_sub_mask = [trg len, trg len]

    #     trg_mask = trg_pad_mask & trg_sub_mask
    #     # trg_mask = [batch size, 1, trg len, trg len]

    #     return trg_mask

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

        embedding_src_enc = self.embedding_enc(src=src) # B S F

        print("SEQ2SEQ: Finish Embedding Encoder----------------------------- \n")

        embedding_src_enc = embedding_src_enc.permute(0, 2, 1) # B F S 

        # enc_src = self.encoder(embedding_src_enc, src_mask)
        enc_src = self.encoder(embedding_src_enc)
        # enc_src = [batch size, src len, hid dim]

        print("SEQ2SEQ: Finish Encoding----------------------------- \n")

        embedding_trg_dec = self.embedding_dec(trg=trg)

        print("SEQ2SEQ: Finish Embedding Decoder----------------------------- \n")

        output, attention = self.decoder(embedding_trg_dec, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        print("SEQ2SEQ: Finish Decoding----------------------------- \n")

        return output, attention

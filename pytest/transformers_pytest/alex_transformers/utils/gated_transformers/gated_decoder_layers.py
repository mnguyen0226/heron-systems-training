# script running the decoder of the Gated Transformers

from typing import Tuple
import torch
import torch.nn as nn

from utils.gated_transformers.encoder_decoder.gated_encoder import (
    Attn,
    LNorm,
    Projection,
    Gate,
)
from utils.gated_transformers.encoder_decoder.gated_decoder import Decoder


class DecoderLayers(nn.Module):
    def __init__(
        self,
        output_dim: int,
        encoder_output_shape: Tuple[int, int],
        n_layers: int,
        nb_heads: int,
        dropout: float,
    ):
        """Decoder class for Gated Transformer which is similar to the Encoder but also has
            + mask multi-head attention layer over target sequence
            + multi-head attention layer which uses the decoder representation as the query
                zand the encoder representation as the key and value

        Parameters
        ----------
        output_dim: int
            input dimension to the Output Embedding Layer
        hid_dim: int
            input hidden dim to the Decoder Layer
        n_layers: int
            number of DecoderLater layers
        nb_heads: int
            number of heads for attention mechanism
        dropout: float
            dropout rate = 0.1
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                Decoder(
                    encoder_output_shape=encoder_output_shape,
                    max_seq_len=1,
                    scale=False,
                    nb_heads=nb_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(
            in_features=encoder_output_shape[0], out_features=output_dim
        )

    def forward(
        self,
        trg: Tuple[int, int],
        enc_src: Tuple[int, int, int],
        # trg_mask: Tuple[int, int, int],
        # src_mask: Tuple[int, int, int],
    ) -> Tuple[tuple, tuple]:
        """Feed-forward of the Decoder contains of preprocess data, DecoderLayer and prediction

        Paramters
        ----------
        trg: [batch size, trg len]
            target token(s)
        enc_src: [batch size, src len, hid dim]
            output from the Encoder
        trg_mask: [batch size, 1, trg len, trg len]
            masked out <pad> of the target token(s)
        src_mask: [batch size, 1, 1, src len]
            masked src but allow to ignore <pad> during training in the tokenized vector since it
                does not provide any value

        Return
        ----------
        output: [batch size, trg len, output dim]
            output embedded, tokenize, positional-encoded vectors of the output
        attention: [batch size, n heads, trg len, src len]
            we will not use this
        """
        for layer in self.layers:
            trg = layer(trg, enc_src)

        trg = trg.permute(0, 2, 1)

        output = self.fc_out(trg)

        return output

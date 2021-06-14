# script running the Encoder of the Gated Transformers

from typing import Tuple
import torch
import torch.nn as nn
from utils.gated_transformers.encoder_decoder.gated_encoder import Encoder


class EncoderLayers(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        n_layers: int,
        nb_heads: int,
        dropout: float,
    ):
        """Encoder class for Gated Transformer which is used for
            + embedding preprocessed input texts
            + calling n_layers EncoderLayers
            + providing encoded output for Decoder

        Parameters
        ----------
        hid_dim: int
            dimension of the output of Input Embedding layer and input to EncoderLayer layer
        n_layers: int
            number of layer(s) of the EncoderLayer
        dropout: float
            dropout rate = 0.1
        """
        super().__init__()
        self.layers = nn.ModuleList(  # This is a wrapper for Gated Encoder
            [
                Encoder(
                    in_shape=in_shape,
                    nb_heads=nb_heads,
                    do_scale=False,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        src: Tuple[int, int],
    ) -> Tuple[int, int, int]:
        """Forwards function for the Encoder

        Parameters
        ----------
        src: [batch_size, src_len]
            tokenized vector input text
        src_mask: [batch_size, 1, 1, src_len]
            masked the input text but allow to ignore <pad> after tokenized during training in the
                tokenized vector since it does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]. This is the dimension that will be maintain till output of Decoder
            position-encoded & embedded output of the encoder layer. The src will be fetched into the Decoder
        """
        for layer in self.layers:
            # src = layer(src, src_mask)
            src = layer(src)

        return src

# Minh's Copy for transformer testing

from collections import OrderedDict
from typing import Tuple

import torch
from adept.network import SubModule2D
from adept.utils.util import DotDict
from torch import nn

from gamebreaker.classifier.network.gated_encoder import Attn
from gamebreaker.classifier.network.gated_encoder import Encoder
from gamebreaker.classifier.network.gated_encoder import Gate
from gamebreaker.classifier.network.gated_encoder import LNorm
from gamebreaker.classifier.network.gated_encoder import Projection


class Decoder(nn.Module):
    def __init__(
        self,
        encoder_output_shape: Tuple[int, int],
        max_seq_len: int,
        nb_heads: int,
        scale: bool,
        dropout: float = 0.0,
    ):
        super().__init__()

        # The shape of the auto-regressive input
        auto_regressive_shape = (encoder_output_shape[0], max_seq_len)

        # This handles the auto-regressive encoding of the decoder
        self.auto_norm = LNorm(auto_regressive_shape)
        self.auto_attn = Attn(auto_regressive_shape, nb_heads, scale, dropout)
        self.auto_gate = Gate(auto_regressive_shape)

        # This section is almost identical to an encoder, but instead of using the output of the
        # previous layers for q, k, and v, we instead use the encoder's to generate k and v
        self.encoder_norm = LNorm(encoder_output_shape)
        self.decoder_norm = LNorm(auto_regressive_shape)
        self.decoder_attn = Attn(encoder_output_shape, nb_heads, scale, dropout)
        self.first_gate = Gate(auto_regressive_shape)

        self.proj_norm = LNorm(auto_regressive_shape)
        self.projection = Projection(auto_regressive_shape, dropout)
        self.second_gate = Gate(auto_regressive_shape)

    def forward(
        self, prev_seq: torch.Tensor, encoder_out: torch.Tensor
    ) -> torch.Tensor:
        """The forward function of the decoder

        Parameters
        ----------
        prev_seq: torch.Tensor
            The previous output of the decoder (or the start-of-sequence token)
        encoder_out: torch.Tensor
            The output from the encoder paired with this decoder
        Returns
        -------
        torch.Tensor
            The decoded sequence
        """
        prev_seq_norm = self.auto_norm(prev_seq)
        prev_seq_attn = self.auto_attn(q_input=prev_seq_norm, kv_input=prev_seq_norm)
        prev_seq_gate = self.auto_gate(prev_seq, prev_seq_attn)

        q_input = self.decoder_norm(prev_seq_gate)
        kv_input = self.encoder_norm(encoder_out)

        attn_out = self.decoder_attn(q_input=q_input, kv_input=kv_input)
        gate_out = self.first_gate(q_input, attn_out)
        gate_norm = self.proj_norm(gate_out)
        proj_out = self.projection(gate_norm)
        proj_gate = self.second_gate(gate_out, proj_out)

        return proj_gate


def run_gated_decoder():
    print("Running run_gated_decoder()")


if __name__ == "__main__":
    run_gated_decoder()

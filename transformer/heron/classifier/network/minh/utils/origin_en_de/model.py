from typing import Tuple

import numpy as np
import torch
from torch import nn

from gamebreaker.classifier.network.gated_decoder import Decoder
from gamebreaker.classifier.network.gated_encoder import Encoder

class GatedTransformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        nb_heads: int,
        max_seq_len: int,
        scale: bool,
        dropout: float,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(input_shape, nb_heads, scale, dropout)
        self.decoder = Decoder(input_shape, max_seq_len, nb_heads, scale, dropout)

    def forward(self, xs: torch.Tensor, prev_seq: torch.Tensor) -> torch.Tensor:
        encoded_xs = self.encoder(xs)
        next_probs = self.decoder(prev_seq, encoded_xs)
        return next_probs


class AutoRegTransformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        nb_heads: int,
        max_seq_len: int,
        scale: bool,
        dropout: float,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(input_shape, nb_heads, scale, dropout)
        self.decoder = Decoder(input_shape, max_seq_len, nb_heads, scale, dropout)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        b, f, s = xs.shape
        encoded_xs = self.encoder(xs)
        outputs = torch.zeros(b, self.max_seq_len, s).to(xs.device)
        prev_seq = torch.zeros(b, f, self.max_seq_len).to(xs.device)
        for t in range(self.max_seq_len):
            new_outputs = self.decoder(prev_seq, encoded_xs)
            selected_indices = torch.argmax(new_outputs, dim=-1)
            prev_seq[list(range(b)), :, t] = xs[list(range(b)), :, selected_indices]
            outputs[:, t, :] = new_outputs

        return outputs


def binary(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Converts x to binary numbers of len(bits)

    Parameters
    ----------
    x: torch.Tensor
        Numbers to convert to binary
    bits: int
        Bit length of output

    Returns
    -------
    torch.Tensor
        Binary representations of x
    """
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def model():
    print("Running model.py")

if __name__ == "__main__":
    model()
from collections import OrderedDict
from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_shape: int, nb_heads: int, do_scale: bool, dropout: float = 0.0):
        """ 
        A single encoding layer for Gated Transformer

        Parameters
        ----------
        in_shape: int

        """

class LNorm(nn.Module):
    def __init__(self, in_shape: int):
        """ 
        Layer norm with added permute/view mechanics

        Parameters
        ----------
        in_shape: int
            The input shape for the module
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        The forward function for the module

        Parameters
        ----------
        x: torch.Tensor
            Input to be passed through the model.
        """
        
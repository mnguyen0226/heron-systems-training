# Minh's copy for transformer testing

from collections import OrderedDict
from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn
from adept.network import SubModule2D
from adept.utils.util import DotDict


class GatedEncoder(SubModule2D):
    args = {}

    def __init__(
        self,
        in_shape: Tuple[int, int],
        id: str = "Gated Encoder",
        nb_encoders: int = 1,
        nb_heads: int = 1,
        do_scale: bool = True,
        dropout: float = 0.0,
    ):
        """
        The encoder layer for a gated transformer, written to be compatible as a net2d module in
        Adept's ModularNetwork

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape for the module. This should be given as [F, S] (no batch). Note that
            this is the only input the user can give on the shape of the network. All sizing is done
            based on this input s.t. the output shape matches the input shape.
        id: str
            Unique identifier for this instance.
        nb_encoders: int
            How many encoding layers to stack together.
        dropout: float
            How often to perform dropout (0.0 for never, 1.0 for always)
        """
        super().__init__(in_shape, id)

        # The output shape of this network is necessarily the input shape.
        self._out_shape = in_shape

        encoder_dict = []
        for i in range(nb_encoders):
            encoder_dict.append(
                (f"encoder_{i}", Encoder(in_shape, nb_heads, do_scale, dropout))
            )

        self.encoders = nn.Sequential(OrderedDict(encoder_dict))

    @classmethod
    def from_args(
        cls, args: DotDict, in_shape: Tuple[int, int], id: str
    ) -> "GatedEncoder":
        """

        Parameters
        ----------
        args: DotDict
            Arguments used to build the model. Must include key "nb_encoders"
        in_shape: Tuple[int, int]
            The input shape of the data (no batch shape). This should be in the order of [F, S],
            where F is the number of features and S is the sequence length. Note that the input
            shape provided will also be the output shape.
        id: str
            Unique identifier for this module

        Returns
        -------
        GatedEncoder
        """
        return cls(in_shape, id, args.nb_encoders, args.nb_heads, args.dropout)

    def _forward(self, xs, internals, **kwargs):
        """Forward function of the module

        Parameters
        ----------
        xs: torch.Tensor
            [B, F, S]
        internals
        kwargs

        Returns
        -------

        """

        encoded_output = self.encoders(xs)

        # We return the encoded output as part of the "internals" just for testing/debugging
        # purposes
        return self.encoders(xs), {"attention_output": encoded_output}

    def _new_internals(self) -> Dict:
        """
        Function for generating internals for an LSTM (unused).

        Returns
        -------
        Dict
            New internals (unused)
        """
        return {}

    @property
    def _output_shape(self) -> Tuple[int, int]:
        """
        The output shape of this module.

        Returns
        -------
        Tuple[int, int]
            The output shape of the module (no batch size)
        """
        return self._out_shape


class Encoder(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        nb_heads: int,
        do_scale: bool,
        dropout: float = 0.0,
    ):
        """
        A single encoding layer for the Gated transformer

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape for the encoding layer. Should be ordered as [F, S] (no batch)
        dropout: float
            The dropout to use for the model after the attn layers and the projection layers
        """
        super().__init__()
        self.first_norm = LNorm(in_shape)
        self.attn_layer = Attn(in_shape, nb_heads, do_scale, dropout)
        self.first_gate = Gate(in_shape)
        self.second_norm = LNorm(in_shape)
        self.projection = Projection(in_shape, dropout)
        self.second_gate = Gate(in_shape)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        The forward function for the encoding layer

        Parameters
        ----------
        xs: torch.Tensor
            Input to be passed through the network. Should have dimensions [B, F, S].

        Returns
        -------
        torch.Tensor
            Output of the encoder with shape [B, F, S] (unchanged from input)
        """

        xs_norm = self.first_norm(xs)
        attn_out = self.attn_layer(xs_norm, xs_norm)
        gate_out = self.first_gate(xs, attn_out)
        gate_norm = self.second_norm(gate_out)
        proj_out = self.projection(gate_norm)
        proj_gate = self.second_gate(gate_out, proj_out)

        return proj_gate


class LNorm(nn.Module):
    def __init__(self, in_shape: Tuple[int, int]):
        """
        Layer norm with added permute/view mechanics

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape for the module. Should be ordered as [F, S] (no batch)
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward function for the module

        Parameters
        ----------
        x: torch.Tensor
            Input to be passed through the model. Should have shape [B, F, S]

        Returns
        -------
        torch.Tensor
            The output of the layer norm. Will have shape [B, F, S] (unchanged from input)
        """
        b, f, s = x.shape

        # We need to re-order the input to be just a really long list of feature vectors, so we
        # combine the batch and sequence
        x_permuted = torch.reshape(x.permute(0, 2, 1), (b * s, f))
        x_norm = self.layer_norm(x_permuted)

        # Return the output to the original shape
        return x_norm.view(b, s, f).permute(0, 2, 1)


class Attn(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int],
        nb_heads: int,
        scale: bool,
        dropout: float = 0.0,
    ):
        """Multi-head self attention layer of the transformer

        Parameters
        ----------
        in_shape: Tuple[int, int]
            Input shape to the attention layer, with dimensions [Features, Sequence]. Sequence is
            unused.
        nb_heads: int
            How many heads to use for multi-head attention
        scale: bool
            Whether or not to include C in the equation sigmoid(q*k^T / C)
        dropout: float
            Probability of dropout (0.0 -> no dropout)
        """
        super().__init__()
        nb_features, _ = in_shape
        self._h = nb_heads
        self.mq = nn.Linear(nb_features, nb_features * nb_heads, bias=False)
        self.mk = nn.Linear(nb_features, nb_features * nb_heads, bias=False)
        self.mv = nn.Linear(nb_features, nb_features * nb_heads, bias=False)

        self.projection_output = nn.Sequential(
            OrderedDict(
                [
                    ("mZ", nn.Linear(nb_features * nb_heads, nb_features, bias=False)),
                    ("relu", nn.ReLU()),
                    ("dropout", nn.Dropout(p=dropout)),
                ]
            )
        )
        self.scale = nb_features ** 0.5 if scale else 1.0

    def forward(self, q_input: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        """Forward function for the attention layer

        Parameters
        ----------
        q_input: torch.Tensor
            Input to be multiplied against the Q matrix.
        kv_input: torch.Tensor
            Input to be multiplied against the K and V matricies. For the encoder case, this should
            be the same as q_input. For the decoder case, this should be the same as q_input for the
            auto-regressive head, but this should be the output of the encoders for the attention
            layer after the autoregressive head

        Returns
        -------
        torch.Tensor
            The weighted attention.
        """
        b, f, q_s = q_input.shape
        _, _, kv_s = kv_input.shape

        q_input = q_input.permute(0, 2, 1).view(b * q_s, f)
        k_input = torch.clone(kv_input).permute(0, 2, 1).view(b * kv_s, f)
        v_input = torch.clone(kv_input).permute(0, 2, 1).view(b * kv_s, f)

        q = self.mq(q_input).view(b, q_s, f, self._h)
        k = self.mk(k_input).view(b, kv_s, f, self._h)
        v = self.mv(v_input).view(b, kv_s, f, self._h)

        qkt = torch.einsum("bifh,bjfh->bijh", q, k)

        attn_weights = (qkt * self.scale).sigmoid()
        weighted_v = torch.einsum("bijh,bjfh->bifh", attn_weights, v)
        weighted_v = weighted_v.reshape(b * q_s, f * self._h)

        output = self.projection_output(weighted_v).view(b, q_s, f).permute(0, 2, 1)

        return output


class Gate(nn.Module):
    def __init__(self, in_shape: Tuple[int, int]):
        """
        Gating layer for the Gated Transformer. Uses a GRU to compare the output of a previous layer
        to the input of a different layer.

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape to the module. Should be ordered [F, S] (no  batch)
        """
        super().__init__()
        self.gru = nn.GRU(input_size=in_shape[0], hidden_size=in_shape[0])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward function for the gating layer

        Parameters
        ----------
        x: torch.Tensor
            This should have shape [B, F, S]. This is the original input to the attention layer
        y: torch.Tensor
            This should have shape [B, F, S]. This is the output from the attention layer

        Returns
        -------
        torch.Tensor
            The output of the gating layer with shape [B, F, S]
        """

        b, f, s = x.shape

        # Permute the x and y so that the shape is now [B, S, F]
        x_permuted = x.permute(0, 2, 1)
        y_permuted = y.permute(0, 2, 1)

        # We really just need the GRU to weight between the input and the self attention. So we
        # resize to be [B * S, 1, F] so that we essentially have a massive batch of samples with
        # sequence length 1 and F features
        gate_output, _ = self.gru(
            torch.reshape(y_permuted, (1, b * s, f)),
            torch.reshape(x_permuted, (1, b * s, f)).contiguous(),
        )

        # Return the output to the original shape
        return gate_output.view(b, s, f).permute(0, 2, 1)


class Projection(nn.Module):
    def __init__(self, in_shape: Tuple[int, int], dropout: float = 0.0):
        """
        The projection layer for the Transformer. Really, just a linear layer with some reshapings

        Parameters
        ----------
        in_shape: Tuple[int, int]
            The input shape to the layer, given as [F, S]. This will determine the number of hidden
            units as well as the output shape of the layer.
        dropout: float
            The dropout to use for the model
        """
        super().__init__()
        self.projection = nn.Sequential(
            OrderedDict(
                [
                    ("projection", nn.Linear(in_shape[0], in_shape[0], bias=False)),
                    ("proj_relu", nn.ReLU()),
                    ("dropout", nn.Dropout(p=dropout)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward function for the projection layer.

        Parameters
        ----------
        x: torch.Tensor
            The input to be passed through. Dimensions are [B, F, S].

        Returns
        -------
        torch.Tensor
            Shape [B, F, S]
        """

        b, f, s = x.shape

        # Change the order of dimensions to be [B, S, F]
        x_permuted = x.permute(0, 2, 1)

        # Project each feature vector separately
        reshaped_x = torch.reshape(x_permuted, (b * s, f))
        y = self.projection(reshaped_x)

        # Return the output in the same shape as the input
        return y.view(b, s, f).permute(0, 2, 1)


def run_gated_encoder():
    print("Running run_gated_encoder()")


if __name__ == "__main__":
    run_gated_encoder()

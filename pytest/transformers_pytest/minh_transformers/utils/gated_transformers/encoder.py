# script running the Encoder of the Gated Transformers

from typing import Tuple
import torch
import torch.nn as nn
from utils.preprocess import device

class Encoder(nn.Module):
    def __init__(
        self, in_shape: Tuple[int, int], n_layers: int, n_heads: int, dropout: float,
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
        self.layers = nn.ModuleList(
            [GatedEncoderLayer(in_shape=in_shape, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(
        self, src: Tuple[int, int], src_mask: Tuple[int, int, int, int]
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
            src = layer(src, src_mask)

        return src


class GatedEncoderLayer(nn.Module):
    def __init__(
        self,
        in_shape: Tuple[int, int],  # hid_dim, 1, F,S (but we will never use sequence)
        n_heads: int,
        dropout: float = 0.0,
    ):
        """Gated Encoder layer of Encoder of the Transformer

        Parameters
        ----------
        hid_dim: int
            input hidden_dim from the processed positioned-encoded & embedded vectorized input text
        n_heads: int
            number of head(s) for attention mechanism
        dropout: float
            dropout rate = 0.1
        """
        super().__init__()
        self.first_norm = LNorm(in_shape=in_shape)
        self.attn_layer = Attn(in_shape, n_heads, dropout)
        self.first_gate = Gate(in_shape=in_shape)

        self.second_norm = LNorm(in_shape=in_shape)
        self.projection = Projection(in_shape, dropout)
        self.second_gate = Gate(in_shape=in_shape)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: Tuple[int, int, int], src_mask: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Feed-forward layer for the Gate Encoder layer

        Parameters
        ----------
        src: [batch size, src len, hid dim] B,S,F
            tokenized vector input text
        src_mask: [batch_size, 1, 1, src_len]
            masked the input text but allow to ignore <pad> after tokenized during training in the
                tokenized vector since it does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]. This is the dimension that will be maintain till output of Decoder
            position-encoded & embedded output of the encoder layer. The src will be fetched into the Decoder
        """
        # first layer norm - already dropped out from Encoder class
        src = self.first_norm(src)

        # self-attention
        _src, _ = self.attn_layer(query=src, key=src, value=src, mask=src_mask)

        first_gate_output, _ = self.first_gate(
            self.dropout(_src), src
        )  # [batch size, src len, hid dim]

        # second layer norm - already dropped from first gate
        src = self.second_norm(first_gate_output)

        # positionwise feedforward
        _src = self.projection(src)

        # second gate
        second_gate_output, _ = self.second_gate(
            self.dropout(_src), src
        )  # [batch size, src len, hid dim]

        return second_gate_output


class LNorm(nn.Module):
    def __init__(self, in_shape: Tuple[int, int]):  # = [hidden_dim, 1]
        """Layer Normalization for both Encoder & Decoder
        This takes the same input as after the Embedding layer

        Parameters
        ----------
        normalized_shape: int
            input shape (hid_dim) of the Encoder and Decoder
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape=in_shape[0]
        )

    def forward(self, x: Tuple[int, int, int]) -> Tuple[int, int, int]:  # use the layer
        """Feed-forward function of the Layer Normalization function

        Parameters
        ----------
        x: [batch size, src len, hid dim] # batch, sequence, frequency
            input dimension (hid_dim) of the Layer Normalization of the Encoder & Decoder
        """
        x = self.layer_norm(x)
        return x


class Attn(nn.Module):
    def __init__(
        self,
        # hid_dim: int,
        in_shape: Tuple[int, int],
        n_heads: int,
        dropout: float = 0.0,
    ):
        """Multi/single Head Attention Layer. This layer define Q,K,V of the GateEncoderLayer

        Parameters
        ----------
        hid_dim: int
            input hidden dimension from the first layer norm
        n_heads: int
            number of heads for attention mechanism
        dropout: float
            dropout rate = 0.1
        """
        super().__init__()

        nb_features, _ = in_shape

        assert (
            nb_features % n_heads == 0
        )  # make sure that number of multiheads are concatenatable

        self.hid_dim, _ = in_shape

        self._h = n_heads
        self.head_dim = nb_features // n_heads  # determine the head_dim

        self.mq = nn.Linear(
            in_features=nb_features, out_features=nb_features, bias=False
        )
        self.mk = nn.Linear(
            in_features=nb_features, out_features=nb_features, bias=False
        )  # apply linear transformation
        self.mv = nn.Linear(
            in_features=nb_features, out_features=nb_features, bias=False
        )  # apply linear transformation

        self.mZ = nn.Linear(
            in_features=nb_features, out_features=nb_features, bias=False
        )  # apply linear transformation

        self.dropout = nn.Dropout(dropout)
        self.scale = (
            nb_features ** 0.5
        )  # Alex's implementation: nb_features ** 0.5 if scale else 1.0

    def forward(
        self,
        query: Tuple[int, int, int],
        key: Tuple[int, int, int],
        value: Tuple[int, int, int],
        mask=None,
    ) -> Tuple[tuple, tuple]:
        """Feed-forward layer for the attention mechanism

        Parameters
        ----------
        query, key, value:
            Query is used with Key to get an attention vector which is then weighted sum with Value
            query: [batch size, query len, hid dim]
            key: [batch size, key len, hid dim]
            value: [batch size, value len, hid dim]
        mask:
            masked the input text but allow to ignore <pad> after tokenized during training in the
                tokenized vector since it does not provide any value

        Return
        ----------
        x: [batch size, query len, hid dim]
            input to the first gate layer
        """
        batch_size = query.shape[0]

        Q = self.mq(
            query
        )  # applied linear transformation but keep dim, Q = [batch size, query len, hid dim]
        K = self.mk(
            key
        )  # applied linear transformation but keep dim, K = [batch size, key len, hid dim]
        V = self.mv(
            value
        )  # applied linear transformation but keep dim, V = [batch size, value len, hid dim]

        # Change the shape of QKV
        Q = Q.view(batch_size, -1, self._h, self.head_dim).permute(
            0, 2, 1, 3
        )  # Q = [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self._h, self.head_dim).permute(
            0, 2, 1, 3
        )  # K = [batch size, n heads, key len, head dim]
        V = V.view(batch_size, -1, self._h, self.head_dim).permute(
            0, 2, 1, 3
        )  # V = [batch size, n heads, value len, head dim]

        # -------------------------------------------------------
        # energy = [batch size, n heads, query len, key len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # matmul, scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)  # mask

        # attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim=-1)  # attention = softmax of QK/d_k

        # x = [batch size, n heads, query len, head dim] # original Q,K,V dim
        x = torch.matmul(self.dropout(attention), V)  # matmul
        # -------------------------------------------------------

        # x = [batch size, query len, n heads, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # Change the shape again for concat

        # x = [batch size, query len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)  # combine the heads together

        # x = [batch size, query len, hid dim]
        x = self.mZ(x)  # Linear layer output for attention

        return x, attention


class Gate(nn.Module):
    def __init__(self, in_shape: Tuple[int, int]):  # F, S = hid_dim,
        """Gate Layer for both the Encoder & Decoder

        Parameters
        ----------
        hid_dim: int
            input hidden dimension (of the Encoder & Decoder)
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_shape[0], hidden_size=in_shape[0]
        ) 

    def forward(
        self, output: Tuple[int, int, int], original_input: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Feed-forward function of the Gate Layer

        Parameters
        ----------
        output: [batch size, src len, hid dim]
            the output from either Attention Layer or Positionwise Layer

        original_input.shape: [batch size, src len, hid dim]
            the input preprocessed text tokens
        """
        b, f, s = original_input.shape  # Split the input shape

        # Permute the x and y so that the shape is now [B,S,F]
        original_input_permuted = original_input.permute(0, 2, 1)
        output_permuted = output.permute(0, 2, 1)

        # We really just need the GRU to weight between the input and the self attention. So we
        # resize to be [B * S, 1, F] so that we essentially have a massive batch of samples with
        # sequence length 1 and F features
        gate_output, hidden = self.gru(
            torch.reshape(output_permuted, (1, b * f, s)),
            torch.reshape(original_input_permuted, (1, b * f, s)).contiguous(),
        )

        return gate_output.view(b, s, f).permute(0, 2, 1), hidden


class Projection(nn.Module):  # I can specify the pf dim
    # def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
    def __init__(self, in_shape: Tuple[int, int], dropout: float):
        """Positionwise Feedforward layer of GatedEncoderLayer
        Why is this used? Unfortunately, it is never explained in the paper.
        The transformed from hid_dim to pf_dim (pf_dim >> hid_dim.
        The ReLU activation function and dropout are applied before it is transformed back into
            hid_dim representation

        Parameters
        ----------
        hid_dim: int
            input hidden dimension from the second layer norm
        dropout: Float
            dropout rate = 0.1
        """
        super().__init__()
        self.fc_1 = nn.Linear(
            in_features=in_shape[0], out_features=in_shape[0], bias=False
        )  # linear transformation
        self.fc_2 = nn.Linear(
            in_features=in_shape[0], out_features=in_shape[0], bias=False
        )  # linear transformation # make sure to conert back from pf_dim to hid_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Feedforward function for the PFF layer

        Parameters
        ----------
        x: [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
            input from the second layer norm

        Return
        ----------
        x: [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
            output to the second gate layer
        """
        # x = [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
        x = self.dropout(
            torch.relu(self.fc_1(x))
        )  # relu then dropout to contain same infor

        # x = [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
        x = self.fc_2(x)

        return x

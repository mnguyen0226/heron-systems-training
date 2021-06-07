# script running the Encoder of the Original Transformers

from typing import Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        device: str,
        max_length=100,
    ):
        """Encoder wrapper for Transformer: preprocessing the input data, call EncoderLayer,
            and provide output

        Parameters
        ----------
        input_dim: int
            input dim of the word vector, not to the EncoderLayer
        hid_dim: int
            dim of the input to the EncoderLayer
        n_layers: int
            number of layers of the EncoderLayer
        n_heads: int
            number of heads of the Attention
        pf_dim: int
            feed_forward input dim?
        dropout: float
            dropout rate = 0.1
        device: str
            CPU or GPU
        max_length: int
            position embedding has a vocab size of 100, which means out model can accept
            sentences up to 100 tokens long.
        """
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hid_dim
        )  # input, output
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=hid_dim
        )  # input, output

        # this is submodule that can be repeat 6 times
        self.layers = nn.ModuleList(
            [EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(
            device
        )  # sqrt(d_model). This is a hidden dim size.

    def forward(
        self, src: Tuple[int, int], src_mask: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """Feed-forward function of Encoder

        Parameters
        ----------
        src: [batch_size, src_len]
            src tokenized input SRC_PAD_IDX
        src_mask: [batch_size, 1, 1, src_len]
            masked src but allow to ignore <pad> during training in the tokenized vector
            since it does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]
            position-encoded & embedded output of the encoder layer. This will be fetch into the decoder
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional vector
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch_size, src_len]

        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )  # have to dropout to get the same dim in the Encoder Layer
        # src = [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch_size, src_len, hid_dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: str):
        """EncoderLayer of the Encoder of Transformer contains Multi-Head Attention, Add&Normal,
            Feed-forward, Add&Norm

        Parameters
        ----------
        hid_dim: int
            input hidden dim from the processed positioned-encoded & embedded vectorized input
        n_heads: int
            number of heads for the attention mechanism
        pf_dim: int
            input feed-forward dim
        dropout: float
            dropout rate
        device: str
            cpu or gpu
        """
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(
            hid_dim
        )  # initialized the norm for attn, dim reserved
        self.ff_layer_norm = nn.LayerNorm(
            hid_dim
        )  # initialized the norm for feed forward, dim reserved
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)  # dropout rate 0.1 for Encoder

    def forward(
        self, src: Tuple[int, int], src_mask: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """Feed-forward layer for then Encoder Layer

        Parameters
        ----------
        src: [batch size, src len, hid dim]
            src tokenized input SRC_PAD_IDX
        src_mask: [batch_size, 1, 1, src_len]
            masked src but allow to ignore <pad> during training in the tokenized vector since it
            does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]
            position-encoded & embedded output of the encoder layer. This will be fetch into the decoder
        """
        _src, _ = self.self_attention(
            query=src, key=src, value=src, mask=src_mask
        )  # not using the attention result

        # dropout, add residual connection and layer norm
        src = self.self_attn_layer_norm(
            src + self.dropout(_src)
        )  # have to dropout _src to have the same rate as src
        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)  # update input

        # dropout, add residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: str):
        """Multi/single Head Attention Layer. Define Q,K,V of the EncoderLayer
        In terms of a single Scaled Dot-Product Attention:
            Q & K is matmuled
            The result is then scaled
            It is then masked
            The result is then softmaxed
            The result is then matmuled with V

        Parameters
        ----------
        hid_dim: int
            input hidden dim to the EncoderLayer
        n_heads: int
            number of heads for attention mechanism
        drouput: Float
            dropout rate
        device: String
            CPU or GPU
        """
        super().__init__()

        assert hid_dim % n_heads == 0  # make sure that multi head is concatenatable

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads  # determine the head_dim

        self.fc_q = nn.Linear(
            in_features=hid_dim, out_features=hid_dim
        )  # apply linear transformation
        self.fc_k = nn.Linear(
            in_features=hid_dim, out_features=hid_dim
        )  # apply linear transformation
        self.fc_v = nn.Linear(
            in_features=hid_dim, out_features=hid_dim
        )  # apply linear transformation

        self.fc_o = nn.Linear(
            in_features=hid_dim, out_features=hid_dim
        )  # apply linear transformation

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(
            device
        )  # d_k = head_dim, not just hid_dim anymore

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
            Is used with key to get an attention vector which is then weighted sum with value
            query = [batch size, query len, hid dim]
            key = [batch size, key len, hid dim]
            value = [batch size, value len, hid dim]
        mask:
            src_mask - masked src but allow to ignore <pad> during training in the tokenized
            vector since it does not provide any value

        Return
        ----------
        src: [batch size, query len, hid dim]
            basically either input to Add&Normalized layer
        attention: int
            attention matrix
        """
        batch_size = query.shape[0]

        Q = self.fc_q(query)  # applied linear transformation but keep dim
        K = self.fc_k(key)  # applied linear transformation but keep dim
        V = self.fc_v(value)  # applied linear transformation but keep dim

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        # Change the shape of QKV
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # -------------------------------------------------------
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # matmul, scale
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)  # mask

        attention = torch.softmax(energy, dim=-1)  # attention = softmax of QK/d_k
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)  # matmull

        # x = [batch size, n heads, query len, head dim] # original Q,K,V dim
        # -------------------------------------------------------

        x = x.permute(0, 2, 1, 3).contiguous()  # Change the shape again for concat
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)  # combine the heads together
        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)  # Linear layer output for attention
        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        """Positionwise Feedforward layer of the EncoderLayer.
        Why is this used? Unfortunately, it is never explained in the paper.
        The transformed from hid_dim to pf_dim (pf_dim >> hid_dim.
        The ReLU activation function and dropout are applied before it is transformed back
            into hid_dim representation

        Parameters
        ----------
        hid_dim: int
            input hidden dim from the Add&Norm Layer
        pf_dim: int
            output feedforward dim
        dropout: float
            dropout rate: 0.1 for encoder
        """
        super().__init__()
        self.fc_1 = nn.Linear(in_features=hid_dim, out_features=pf_dim)  # linear transformation
        self.fc_2 = nn.Linear(
            in_features=pf_dim, out_features=hid_dim
        )  # linear transformation # make sure to conert back from pf_dim to hid_dim

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Feedforward function for the PFF Layer

        Parameters
        ----------
        x: [batch size, seq len, hid dim]
            input from the Add&Norm Layer

        Return
        ----------
        x: [batch size, seq len, hid dim]
            output to Add&Norm Layer
        """
        x = self.dropout(torch.relu(self.fc_1(x)))  # relu then dropout to contain same infor
        # x = [batch size, seq len, pf dim] OR [batch size, src len, hid dim]

        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim] OR [batch size, src len, hid dim]

        return x

# script running the decoder of the Original Transformers

from typing import Tuple
import torch
import torch.nn as nn

from utils.original_transformers.encoder import MultiHeadAttentionLayer
from utils.original_transformers.encoder import PositionwiseFeedforwardLayer


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        device: str,
        max_length=100,
    ):
        """Decoder wrapper takes the conded representation of the source sentence Z and convert it
            into predicted tokens in the target sentence.
        Then compare the target sentence with the actual tokens in thetarge sentence to calculate the loss
            which will be used to calculated the gradients of parameters. Then use the optimizer to
                update the weight to improve the prediction.

        The Decoder is similar to encoder, however, it now has 2 multi-head attention layers.
        - masked multi-head attention layer over target sequence
        - multi-head attention layer which uses the decoder representation as the query and the encoder
            representation as the key and value

        Parameters
        ----------
        output_dim: int
            input to the Output Embedding Layer
        hid_dim: int
            input hidden dim to the Decoder Layer
        n_layers: int
            number of DecoderLayer layers
        n_heads: int
            number of heads for attention mechanism
        pf_dim: int
            output fim of the feed-forward layer
        dropout: float
            dropout rate = 0.1
        device: str
            cpu or gpu
        max_length: int
            the positional encoding have a vocab of 100 meaning that they can accept sequences
                up to 100 tokens long
        """
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=hid_dim
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=hid_dim
        )

        # DecoderLayer
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )

        # Linear of the output
        self.fc_out = nn.Linear(hid_dim, output_dim)

        # Softmax the output
        self.dropout = nn.Dropout(dropout)

        # d_k
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(
        self,
        trg: Tuple[int, int],
        enc_src: Tuple[int, int, int],
        trg_mask: Tuple[int, int, int],
        src_mask: Tuple[int, int, int],
    ) -> Tuple[tuple, tuple]:
        """Feed-forward of the Decoder contains of preprocess data, DecoderLayer and prediction

        Paramters
        ----------
        trg: [batch size, trg len]
            target token(s)
        enc_src: [batch size, src len, hid dim] -> same dim from the output of Encoder
            output from the Encoder
        trg_mask: [batch size, 1, trg len, trg len]
            masked out <pad> of the target token(s)
        src_mask: [batch size, 1, 1, src len]
            masked src but allow to ignore <pad> during training in the tokenized vector
                since it does not provide any value

        Return
        ----------
        output: [batch size, trg len, output dim]
            output embedded, tokenize, positional-encoded vectors of the output
        attention: [batch size, n heads, trg len, src len]
            we will not use this
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        # pos = [batch size, trg len]

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )
        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(
        self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: str
    ):
        """DecoderLayer for the Decoder which contains of:
            + Masked Multi-Head Attention - "self-attention"
            + Add&Norm
            + Multi-Head Attention - "encoder-attention"
            + Add&Norm
            + Feed Forward
            + Add&Norm

        Self-attention layer use decoder's representation as Q,V,K similar as the EncoderLayer.
            Then it follow the Add&Norm which is dropout, residual/adding connection then normalization
            This layer uses the target sequence mask "trg_mask" in order to prevent the decoder from
                cheating by paying attention to tokens
            that are "ahead" of one it is currently processing as it processes all tokens in the target
                sentence in paralel

        Encoder-attention used by feeding the encoded source sentence "enc_src". Q from Decoder and V, K from Encoder.
            The src_mask is used to prevent the multi head attention layer from attending to <pad>
                tokens within the source sentence.
            This is the followed by the Add&Norm (dropout, residual connection, and layer normalization layer)

        The we pass the result to the position-wise feedforward layer and another Add&Norm (dropout, residual
            connection adn layer normalization)

        Parameters
        ----------
        hid_dim: int
            input dim for the DecoderLayer
        n_heads: int
            number of heads for the attention mechanism
        pf_dim: int
            output dim for the feed-forward layer
        dropout: float
            dropout rate = 0.1
        device: str
            cpu or gpu
        """
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(normalized_shape=hid_dim)  # add&norm 1
        self.enc_attn_layer_norm = nn.LayerNorm(normalized_shape=hid_dim)  # add&norm 2
        self.ff_layer_norm = nn.LayerNorm(normalized_shape=hid_dim)  # add&norm 3
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device
        )  # masked multi-head attention
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device
        )  # multi-head attention with encoder
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )  # feed-forward layer

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg: Tuple[int, int, int],
        enc_src: Tuple[int, int, int],
        trg_mask: Tuple[int, int, int, int],
        src_mask: Tuple[int, int, int, int],
    ) -> Tuple[tuple, tuple]:
        """Feed-forward layer for the DecoderLayer with order:
            + Masked Multi-Head Attention
            + Add&Norm
            + Multi-Head Attention
            + Add&Norm
            + Feed-forward
            + Add&Norm

        Parameters
        ----------
        trg: [batch size, trg len, hid dim]
            target token(s)
        enc_src: [batch size, src len, hid dim]
            encoder_source - the output from Encoder
        trg_mask: [batch size, 1, trg len, trg len]
            target mask to prevent the decoder from "cheating" by paying attention to tokens that are "ahead"
                of the one it is currently processing as it processes all tokens in the target sentence in parallel
        src_mask: [batch size, 1, 1, src len]
            source mask is used to prevent the multi-head attention layer from attending to <pad> tokens within
                the source sentence.

        Return
        ----------
        trg: [batch size, trg len, hid dim]
            the predicted token(s)
        attention: [batch size, n heads, trg len, src len]
            We will not use this for our case
        """
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm (Add&Norm)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))  # update new target
        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention

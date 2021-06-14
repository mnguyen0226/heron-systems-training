# script running the decoder of the Gated Transformers

from typing import Tuple
import torch
import torch.nn as nn
from utils.preprocess import device

from utils.gated_transformers.encoder import Attn
from utils.gated_transformers.encoder import LNorm
from utils.gated_transformers.encoder import Projection
from utils.gated_transformers.encoder import Gate


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

        # linear layer of the output
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
            # trg = [batch size, trg len, hid dim]
            # attention = [batch size, n heads, trg len, src len]
            # trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            # trg, attention = layer(trg, enc_src)
            trg = layer(trg, enc_src)
            # print(f"DECODER: size of processed target {trg.shape}")

        trg = trg.permute(0, 2, 1)
        # print(f"DECODER: size of processed target {trg.shape}")

        # output = [batch size, trg len, output dim]
        # print(f"DECODER: size of processed target {trg.shape}")
        output = self.fc_out(trg)
        # print(f"DECODER: size of processed output {output.shape}")

        # return output, attention
        return output


class Decoder(nn.Module):
    def __init__(
        self, 
        encoder_output_shape: Tuple[int, int], # hid_dim, 1
        max_seq_len: int,
        scale: bool, 
        nb_heads: int, 
        dropout: float
    ):
        """Gated Decoder Layer for the Decoder

        Self-attention layer use decoder's representation as Q,V,K similar as the EncoderLayer.
            Then it follow the Add&Norm which is dropout, residual/adding connection then normalization
            This layer uses the target sequence mask "trg_mask" in order to prevent the decoder from
                cheating by paying attention to tokens
            that are "ahead" of one it is currently processing as it processes all tokens in the
                target sentence in paralel

        Encoder-attention used by feeding the encoded source sentence "enc_src". Q from Decoder and
            V, K from Encoder.
            The src_mask is used to prevent the multi head attention layer from attending to <pad>
                tokens within the source sentence.
            This is the followed by the Add&Norm (dropout, residual connection, and layer normalization layer)

        Parameters
        ----------
        hid_dim: int
            input hidden_dim from the processed positioned-encoded & embedded vectorized input text
        nb_heads: int
            number of head(s) for attention mechanism
        dropout: float
            dropout rate = 0.1
        """
        super().__init__()
        # self.first_layer_norm = LNorm(in_shape=encoder_output_shape)
        # self.self_attention = Attn(encoder_output_shape, nb_heads, dropout)
        # self.first_gate = Gate(in_shape=encoder_output_shape)

        # self.second_layer_norm = LNorm(in_shape=encoder_output_shape)
        # self.encoder_attention = Attn(encoder_output_shape, nb_heads, dropout)
        # self.second_gate = Gate(in_shape=encoder_output_shape)

        # self.third_layer_norm = LNorm(in_shape=encoder_output_shape)
        # self.positionwise_feedforward = Projection(encoder_output_shape, dropout)
        # self.third_gate = Gate(in_shape=encoder_output_shape)

        # self.dropout = nn.Dropout(dropout)
        
        #################################
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

    # def forward(
    #     self,
    #     trg: Tuple[int, int, int],
    #     enc_src: Tuple[int, int, int],
    #     trg_mask: Tuple[int, int, int, int],
    #     src_mask: Tuple[int, int, int, int],
    # ) -> Tuple[tuple, tuple]:
    #     """Feed-forward layer for the Gated Decoder

    #     Parameters
    #     ----------
    #     trg: [batch size, trg len, hid dim]
    #         target token(s)
    #     enc_src: [batch size, src len, hid dim]
    #         encoder_source - the output from Encoder
    #     trg_mask: [batch size, 1, trg len, trg len]
    #         target mask to prevent the decoder from "cheating" by paying attention to tokens that are
    #             "ahead" of the one it is currently processing as it processes all tokens in the target
    #                 sentence in parallel
    #     src_mask: [batch size, 1, 1, src len]
    #         source mask is used to prevent the multi-head attention layer from attending to <pad>
    #             tokens within the source sentence.

    #     Return
    #     ----------
    #     trg: [batch size, trg len, hid dim]
    #         the predicted token(s)
    #     attention: [batch size, n heads, trg len, src len]
    #         We will not use this for our case
    #     """
    #     # first layer norm - already dropped out from the Decoder class
    #     trg = self.first_layer_norm(trg)

    #     # self-attention
    #     _trg, _ = self.self_attention(
    #         query=trg, key=trg, value=trg, mask=trg_mask
    #     )  # _trg = [batch size, trg len, hid dim]

    #     # first gate
    #     first_gate_output, _ = self.first_gate(self.dropout(_trg), trg)

    #     # second layer norm - already dropped out from the first gate
    #     trg = self.second_layer_norm(first_gate_output)

    #     # encoder-attention
    #     _trg, attention = self.encoder_attention(
    #         query=trg, key=enc_src, value=enc_src, mask=src_mask
    #     )  # _trg = [batch size, trg len, hid dim]

    #     # second gate
    #     second_gate_output, _ = self.second_gate(self.dropout(_trg), trg)

    #     # third layer norm - already dropped out from the second gate
    #     trg = self.third_layer_norm(
    #         second_gate_output
    #     )  # _trg = [batch size, trg len, hid dim]

    #     # positionwise feedforward
    #     _trg = self.positionwise_feedforward(trg)

    #     # third gate
    #     third_gate_output, _ = self.third_gate(self.dropout(_trg), trg)

    #     return third_gate_output, attention

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
        # print(f"DECODER input target: {prev_seq.shape}\n")

        prev_seq_norm = self.auto_norm(prev_seq) # layer norm take in the target
        # print(f"DECODER first norm {prev_seq_norm.shape}")
        prev_seq_attn = self.auto_attn(q_input=prev_seq_norm, kv_input=prev_seq_norm) # attention of the target
        # print(f"DECODER attn {prev_seq_attn.shape}")
        prev_seq_gate = self.auto_gate(prev_seq, prev_seq_attn) # gate of the target
        # print(f"DECODER first gate {prev_seq_gate.shape}\n")

        q_input = self.decoder_norm(prev_seq_gate) # layer normalize of the target
        # print(f"DECODER Q Input {q_input.shape}")
        kv_input = self.encoder_norm(encoder_out) # # layer normalize of the encoder output
        # print(f"DECODER KV Input {kv_input.shape}")
        attn_out = self.decoder_attn(q_input=q_input, kv_input=kv_input) # decoder attention output
        gate_out = self.first_gate(q_input, attn_out) # gating
        # print(f"DECODER: attn 2 {attn_out.shape}")
        # print(f"DECODER: gate 2 {gate_out.shape}\n")

        gate_norm = self.proj_norm(gate_out)
        # print(f"DECODER norm 3 {gate_norm.shape}")
        proj_out = self.projection(gate_norm)
        # print(f"DECODER projection {proj_out.shape}")
        proj_gate = self.second_gate(gate_out, proj_out)
        # print(f"DECODER gate 3 {proj_gate.shape}\n")

        return proj_gate
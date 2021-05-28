import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

"""
About: The mode is made up of an encoder and decoder with encoder encode input/source sentence into context vector
    and decoder then decode the context vector to output our output/target sentence

ENCODER:
- Does not attempt to compress the entire source sentence into a singlle context vector, instead, it products a sequence of context vectors.
    Why this is not called a sequence of hidden state? 
        + A hidden state at time t in RNN has only seen tokens x_t and all the tokens before it
        + However, each context vector here has seen all tokens at all positions within the input sequence
- How it works?
    + First, the tokens are passed thru a standard embedding layer     
    + Next, as the model has no recurrent, it has no idea about the order of the tokens within the sequence.
        We solve this by using a second embedding layer called a positional embedding layer
        The position embedding has a vocab size of 100 meaning that the model can accept sentences up to 100 tokens long
        This 100 can increase if we want to handle longer sentence
    + The token and positional embedding are elementwise summed together to get a vector which contains info about the tokens and also its position with in the sequence
    + the combined embeddings are then passed thru N encoder layers to get Z, which is then output and can be used by decoder 
"""


"""
ENCODER LAYER:
- We first pass the src sentence and its mask into the multi-head attention layer, then perform a dropout on it, apply a sedifual connection and pass it thru a Layer Normalization  layer
- We then pass it thru a position-wise feedforward layer and then again apply dropout, a residual connection and then a layer normalization to get the output of this layer which is fed into the next layer.
- The muti-head attention layer is used by encoder layer to attend to the source sentence meaning that it calculate and applies attention over itself instead of another sequence, hence we call it self-attention
- This basically jsut resolve input and call the encoder sublayers but not implement the sublayer yet. It does allow for repetition tho
"""
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length = 100):
        super().__init__()
        self.device = device 
        self.tok_embedding = nn.Embedding(input_dim, hid_dim) # input, output
        self.pos_embedding = nn.Embedding(max_length, hid_dim) # input, output

        # this is submodule that can be repeat 6 times
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch_size, src_len]
        # src_mask = [batch_size, 1,1, src_len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional vector
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # pos = [batch_size, src_len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))

        # src = [batch_size, src_len, hid_dim]
        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch_size, src_len, hid_dim]

        return src

"""
ENCODER LAYER - ENCODER SUBLAYERS
- We first pass the srouce sentence and its mask into the multi-head attention layer, the perform drop out on it.
- We the apply the residual connection and pass thru the layer normalization layer.
- We then pass it thru a position-wise feedforward laer and then apply dropout, a residual connection and a normalization to get the output of this SUBLAYER which will then feed to a next sublayer.

The multihead attention layer is used by the encoder layer to attend to the srouce sentece = it calculating and applying attention over itself intead of another sequence =? self attention
"""
class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 

        #self attention         #_src = [batch size, query len, hid dim]
        _src, _ = self.self_attention(src, src, src, src_mask) # not using the attention result

        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]

        return src

"""
MULTI HEAD ATTENTION LAYER
- Attention can be through as queries, keys, and values 
    + Query is used with the key to get the attention vector (usually the output of the softmax operation and has all values between 0 and 1 which sum to 1) 
        which is then used to get the weighted sum of the values
    + d_k help scale the dot product attention which is used to stop the results of the dot products growing large, causing the gradients to become too small
    + The scaled dot-product. Instead of doing a sigle attention application, the hid_dim split into h heads and the scaled dotproduct attentin is calculated over all heads in parallel.
        This means instead of paying attention to 1 concep per attention application, we pay attention to h.
    + Recombine the heads into their hid_dim shape, thus each hid_dim is potentially paying attentoion to h dif conceps
"""
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0 # make sure that multi head is concatenatable

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device) # d_k

    def forward(self, query, key, value, mask = None):
        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        # Change the shape of QKV
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1) # attention = softmax of QK/d_k

        #attention = [batch size, n heads, query len, key len]
        
        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous() # Change the shape again
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim) # combine the heads together
        
        #x = [batch size, query len, hid dim]

        x = self.fc_o(x) # Linear layer output for attention
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

"""
POSITION WISE FEEDFORWARD LAYER (did not explain in the paper)
- The input is transformed from hid_dim to pf_dim (a lot larger than hid_dim)
- The original Transformer used a hid_dim of 512 and a pf_dim of 2048.
- The ReLU activation function and dropout are applied before it is transformed back into a hid_dim representation
"""
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x
###############################################################################
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention

"""
DECODER LAYER:
- The decoder layer is similar to the encoder layer except that it is not has 2 multi-head attention layer, self_attention and encoder_attention
- The "MASKED MULTIHEAD ATTENTION" performs self-attention - QKV like the encoder. 
    + This followed by dropout, residual connection, and layer normalization.
    + This self-attention layer uses the target senttence mask, trg_mask, in order to prevent the decoder from cheating by pating attention to tokens that are ahead of the one it is currently processing as it process all tokens in the target sentence in parallel
- The second "MULTIHEAD ATTENTION" is how we actually feed the encoded source sentence - enc_src into our decoder. 
    + In this multi-head attention layer, the queries are the decoder representation and the keys K and values V are the encoder representaion
    + Here the source mask, src_mask is used to prevent the multi_head attention layer from attending to <pad> token within th source sentence
    + This is then followed by the dropout, residual connection and layer normalization layer
- We pass this through the position-wise feedforward layer and yet another sequence of dropout, residual connection and layer normalization.
"""
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)

        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention
###############################################################################
"""
SEQ2SEQ model - capture encoder and decoder + create masks

- The src mask is created by checking where the source sequence is not equal to a <pad> token. It is 1 where the token is not a <pad> token and 0 when it is. It is then unsqueezed so it can be correctly broadcast when applying the mask to the energy, which of shape [batch size, n heads, seq len, seq len].
- The trg mask:
    + First, we create a mask for the <pad> tokens, as we did for the source mask.
    + Next, we create a "subsequent" mask, trg_sub_mask, using torch.tril.
        This creates a diagonal matrix where the elements above the diagonal will be zero and the elements below the diagonal will be set to whatever the input tensor is
        This shows what each target token (row) is allowed to look at (column). The first target token has a mask of [1, 0, 0, 0, 0] which means it can only look at the first target token. The second target token has a mask of [1, 1, 0, 0, 0] which it means it can look at both the first and second target tokens.
- After the masks are created, they used with encoder and decoder along with the src and target sentence to get our predicted target sentence, utput, along withthe decoder's attemntion over the src sentence
"""
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
###############################################################################
def model():
    print("Runnning Model")

if __name__ == "__main__":
    model()
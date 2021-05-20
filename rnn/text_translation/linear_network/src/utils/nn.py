import random
import sys

from numpy.core.defchararray import decode
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from utils.data_preprocessing import de_vocab, en_vocab
from utils.data_loader import device

# Class Linear create a linear networks
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device ,bias = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))
        self.device = device

    def forward(self, src: Tensor, trg: Tensor)->Tensor:
        batch_size = src.shape[0]
        max_len = trg.shape[0]
        trg_vocal_size = len(en_vocab)
       
        outputs = torch.zeros(max_len, batch_size, trg_vocal_size).to(self.device)

        # first input to the decoder is the <sos> token
        output = trg[0, :]

        # _,y = src.shape
        # if (y != self.in_features):
        #     sys.exit(f"Wrong Input Features. Please use tensor with {self.in_features} Input Features")

        for t in range(1, max_len):
            print(f"TESTING: {src.shape}") 
            print(f"TESTING: {self.weight.T.shape}")
            print(f"TESTING: {self.bias.shape}")

            output = src @ self.weight.t() + self.bias
            outputs[t] = output
        
        return outputs

    # def forward(self, input):
    #     _, y = input.shape
    #     if (y != self.in_features):
    #         sys.exit(f"Wrong Input Features. Please use tensor with {self.in_features} Input Features")
    #     output = input @ self.weight.T + self.bias
    #     return output


INPUT_DIM = len(de_vocab)
OUTPUT_DIM = len(en_vocab)
print(f"TESTING input dim {INPUT_DIM}")
print(f"TESTING output dim {OUTPUT_DIM}")

model = Linear(in_features=INPUT_DIM, out_features=OUTPUT_DIM, bias=True, device=device).to(device)

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model):,} trainable parameters")

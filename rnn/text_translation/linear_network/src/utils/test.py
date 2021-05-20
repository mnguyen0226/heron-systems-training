import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# Class Linear create a linear networks
class myLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self,input):
        _,y = input.shape
        if (y != self.in_features):
            sys.exit(f"Wrong Input Features. Please use tensor with {self.in_features} Input Features")
        output = input @ self.weight.t() + self.bias
        return output

def main():
    print("Running")
    mln = myLinear(20,10)
    input = torch.randn(5,20)
    mln(input)


if __name__ == "__main__":
    main()
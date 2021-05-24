import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
About: the model is made of encoder and decoder
- Encoder encodes the input sequence, in the source language, into a context vector
- Decoder decodes the context vector to produce the output sentence in the target language 

ENCODER:
- The previous models had encoder that compresses an entire input sentence into a single context vector
- the CS2S is different - it gets 2 context vector for each token in an input sentence
- 2 context vectors per token are conved vector and combined vector.


CONVOLUTION-ENCODER:

IMPLEMENTATION:


DECODER:

CONVOLUTION-DECODER:


IMPLEMENTATION:

"""
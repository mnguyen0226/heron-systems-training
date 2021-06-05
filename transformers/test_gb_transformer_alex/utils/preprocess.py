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

# keep randomize seed to be constant every run
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# generate German & English tokenizers
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def tokenize_de(text):
    """Tokenizes German text from a string into a list of strings

    Parameters
    ----------
    text:
        input text sentence(s)
    """
    return [
        tok.text for tok in spacy_de.tokenizer(text)
    ]  # this is just parsing, not vectorized yet


def tokenize_en(text):
    """Tokenizes English text from a string into a list of strings

    Parameters
    ----------
    text:
        input text sentence(s)
    """
    return [
        tok.text for tok in spacy_en.tokenizer(text)
    ]  # this is just parsing, not vectorized yet


# by default RNN Models in PyTorch require the sentence to be a tensor of shape [sequence length, batch size] to TorchText by default
# will return the batches of tensors in the same shape. However, the Transformer will expect the batch dim to be first.
# we tell TorchText to have batches to be [batch size, sequence length] by setting batch_first
SRC = Field(
    tokenize=tokenize_de,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
    batch_first=True,
)
TRG = Field(
    tokenize=tokenize_en,
    init_token="<sos>",
    eos_token="<eos>",
    lower=True,
    batch_first=True,
)

# load Multi30k dataset, feature(SRC) and label(TRG), then split them into train, valid, test data
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(SRC, TRG)
)

# build vocab by converting any tokens that appear less than 2 times into <unk> tokens. Why?
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# set device either cpu/gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

# define training, validating, testing iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device
)


def test_preprocess():
    print("Running preprocess")


if __name__ == "__main__":
    test_preprocess()

"""
Name: Minh Nguyen
Date: 5/26/2021
Resource: https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb

About: Preprocess data by tokenizing and building vocabulary for German to English Translation task with spaCy
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
from typing import Tuple
import random
import math
import time

# set random seeds fro reproducability
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# load German & English spaCy model for tokenize
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

# create a tokenizer
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# create fields for training sources text and targets/labels
SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True)

TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)

# create train, valid, test datasets
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(SRC, TRG)
)

# build a vocabulary
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create interators
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device
)


def run_preprocess():
    print("Running preprocess.py")


if __name__ == "__main__":
    run_preprocess()

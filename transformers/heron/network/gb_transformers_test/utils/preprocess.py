# script preprocessing the translation German-English dataset for both the original Transformers & the Gated Transformers

import torch
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import spacy
import numpy as np
import random

# keep the same randomized seed to get the same randomization everytime
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# set the device to be cpu or gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# batch size for train, validate, test dataset
BATCH_SIZE = 128

# two dictionaries have to be global
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")


def tokenize_de(text: str) -> list:
    """
    Tokenizes German text from a string into a list of strings

    Parameters
    ----------
    text: str
        input text sentence(s)

    Return
    ----------
    list of tokens: list
    """

    return [
        tok.text for tok in spacy_de.tokenizer(text)
    ]  # this is just parsing, not vectorized yet


def tokenize_en(text: str) -> list:
    """
    Tokenizes English text from a string into a list of strings

    Parameters
    ----------
    text: str
        input text sentence(s)

    Return
    ----------
    list of tokens: list
    """
    return [
        tok.text for tok in spacy_en.tokenizer(text)
    ]  # this is just parsing, not vectorized yet


def create_library():
    """Create SRC and TRG Field

    Note
    ----------
        By default RNN models in Pytorch require the sentence to be a tensor of shape
    [sequence length, batch_size] to torchtext by default  will return the batches of
    tensors in the same shape. However, the Transformer will expect the batch dimension
    to be first
        We tell torchtext to have the batches to be [batch size, sequence length] by
    setting batch_first

    Parameter
    ----------
    tokenize_de:
        proccessed German tokenizer
    tokenize_en:
        proccessed English tokenizer

    Return
    ----------
    SRC:
        processed input source field
    TRG:
        processed input label field
    """
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

    return SRC, TRG


def data_iter(device, batch_size, train_data, valid_data, test_data):
    """Build data iterators to access & train elementwise in the dataset

    Paramater
    ----------
    device:
        cpu or gpu
    batch_size:
        batch size
    load_data:
        loaded dataset

    Return
    ----------
    train_iterator:
        training iteractor
    test_iterator:
        testing iterator
    valid_iterator:
        validating iterator
    """
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size, device=device
    )

    return train_iterator, valid_iterator, test_iterator


def run_preprocess():
    SRC, TRG = create_library()

    # load Multi30k dataset, feature(SRC) and label(TRG), then split them into train, valid, test data
    train_data, valid_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))

    # Build vocabularies by converting any tokens that appear less than 2 times into
    # <unk> tokens
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    train_iterator, valid_iterator, test_iterator = data_iter(
        device=device,
        batch_size=BATCH_SIZE,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    return train_iterator, valid_iterator, test_iterator, SRC, TRG

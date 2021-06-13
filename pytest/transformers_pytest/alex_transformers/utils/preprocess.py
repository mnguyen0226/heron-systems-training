# script preprocessing the translation German-English dataset for both the original Transformers & the Gated Transformers

import torch
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import spacy
import numpy as np
import random
from typing import Tuple
import sys

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(edgeitems=sys.maxsize)


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


def create_library() -> Tuple[list, list]:
    """Create SRC and TRG Field

    Note
    ----------
        By default RNN models in Pytorch require the sentence to be a tensor of shape
    [sequence length, batch_size] to torchtext by default  will return the batches of
    tensors in the same shape. However, the Transformer will expect the batch dimension
    to be first
        We tell torchtext to have the batches to be [batch size, sequence length] by
    setting batch_first

    Return
    ----------
    SRC: list
        processed input source field
    TRG: list
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


def data_iter(
    device: str, batch_size: int, train_data: list, valid_data: list, test_data: list
) -> Tuple[list, list, list]:
    """Build data iterators to access & train elementwise in the dataset

    Paramater
    ----------
    device: str
        cpu or gpu
    batch_size: int
        batch size
    train_data: list
        training dataset from loaded dataset
    valid_data: list
        validating dataset from loaded dataset
    test_data: list
        testing dataset from loaded dataset

    Return
    ----------
    train_iterator: list
        training iteractor
    test_iterator: list
        testing iterator
    valid_iterator: list
        validating iterator
    """
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), batch_size=batch_size, device=device
    )

    return train_iterator, valid_iterator, test_iterator


def run_preprocess() -> Tuple[list, list, list, list, list]:
    """Run data preprocessing process

    Return
    ----------
    train_iterator: list
        training dataset iterator
    valid_iterator: list
        validating dataset iterator
    test_iterator: list
        testing dataset iterator
    """
    SRC, TRG = create_library()

    # load Multi30k dataset, feature(SRC) and label(TRG), then split them into train, valid, test data
    train_data, valid_data, test_data = Multi30k.splits(
        exts=(".de", ".en"), fields=(SRC, TRG)
    )

    # used to validates the training, validating, and testing
    print(f"Number of training examples: {len(train_data.examples)}")
    print(f"Number of validating examples: {len(valid_data.examples)}")
    print(f"Number of testing examples: {len(test_data.examples)}")

    print(vars(train_data.examples[0]))

    # build vocabularies by converting any tokens that appear less than 2 times into
    # <unk> tokens
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    # used to check the source & target vocab
    print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
    print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

    # create iterator too splite the training, validating, and testing to batch size
    train_iterator, valid_iterator, test_iterator = data_iter(
        device=device,
        batch_size=BATCH_SIZE,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    # the shape is [batch_size, frequency = number of words, sequence = number of elements]
    # we only pass in the neural network [batch_size, frequency]

    return train_iterator, valid_iterator, test_iterator, SRC, TRG


if __name__ == "__main__":

    train_iterator, valid_iterator, test_iterator, SRC, TRG = run_preprocess()

    print(f"Preprocess First batch Train Iterator: {next(iter(train_iterator)).src}")
    print(f"Preprocess First batch Train Iterator: {next(iter(train_iterator)).trg}")

    print(f"Preprocess First batch Validate Iterator: {next(iter(valid_iterator))}")
    print(f"Preprocess First batch Test Iterator: {next(iter(test_iterator))}")
    print(f"Preprocess SRC {SRC}")
    print(f"Preprocess TRG {TRG}")

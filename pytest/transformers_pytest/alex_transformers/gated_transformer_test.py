# pytest script that make sure the final train/validate loss and train/validate PPL of the gated transformers is
# better than the original transformers from "Attention Is All You Need"
# We want the Train Loss, Val Loss, Train PPL, and Val PPL to be lower than the test bench
# RUNNING COMMAND: pytest -s pytest_transformers.py

from decimal import Decimal
from typing import Tuple
import torch
import torch.nn as nn
from utils.preprocess import run_preprocess, device

def preprocess():
    # Initialize iterator, SRC field, and TRG field
    train_iterator, valid_iterator, test_iterator, SRC, TRG = run_preprocess()


if __name__ == "__main__":
    preprocess()
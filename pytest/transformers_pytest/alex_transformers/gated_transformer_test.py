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

    # Initialize variables for Gated Transformers. This can be adjusted
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    GATED_ENC_LAYERS = 3
    GATED_DEC_LAYERS = 3
    GATED_ENC_HEADS = 8
    GATED_DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    # Initializes variables for training process. This can be adjusted
    N_EPOCHS = 10
    CLIP = 1
    LEARNING_RATE = 0.0005

    # Initializes Encoder Layers

    # Initializes Decoder Layers

    # Defines whole Seq2Seq encapsulating mode
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    # Initializes model

    # Initializes model's weights

    # Initializes Adam Optimizer

    # Initializes Cross Entropy Loss Function

    # Variables for printint format
    gated_transformers_enc_layers = GATED_ENC_LAYERS
    gated_transformers_dec_layers = GATED_DEC_LAYERS
    gated_transformers_enc_n_heads = GATED_ENC_HEADS
    gated_transformers_dec_n_heads = GATED_DEC_HEADS

    # results from Gated Transformers
    print(
        "\n------------------------------------------------------------------------------------------"
    )
    print(
        f"\nThe gated Transformer has {gated_transformers_enc_n_heads} encoder head(s), {gated_transformers_dec_n_heads} \
    decoder head(s), {gated_transformers_enc_layers} encoder layer(s), {gated_transformers_dec_layers} decoder layer(s)"
    )

    print("-----------------------------------------")
    print("Traininng Gated Transformers Training Set")
    print("-----------------------------------------")


    print("----------------------------------------")
    print("Traininng Gated Transformers Testing Set")
    print("----------------------------------------")

if __name__ == "__main__":
    preprocess()

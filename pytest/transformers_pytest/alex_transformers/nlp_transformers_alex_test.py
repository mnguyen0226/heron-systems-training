# pytest script that make sure that the final train/validate PPL of actual gated transformers
# we want to see that the gated transformers is better than "Attention Is All You Need" Transformers
# we want the Train Loss, Val Loss, Train PPL, and Val PPL to be lower than the

# goals:
# 1. have the code compile
# 2. compare results with the original
# 3. Tweak the parameters

# this file call Encoder, Decoder, Seq2Seq, Preprocess, and Training Function
#

from decimal import Decimal
from typing import Tuple
import torch
import torch.nn as nn
from utils.preprocess import run_preprocess, device

from utils.original_transformers.training_utils import (
    origin_transformers_main,
    test_origin_transformers_model,
)
from utils.original_transformers.encoder import Encoder
from utils.original_transformers.decoder import Decoder
from utils.original_transformers.seq2seq import Seq2Seq
from utils.original_transformers.training_utils import (
    initialize_weights,
    origin_transformers_main,
    test_origin_transformers_model,
)

from utils.gated_transformers.gated_decoder import *
from utils.gated_transformers.gated_encoder import *
from utils.gated_transformers.gated_seq2seq import *
from utils.gated_transformers.gated_training_utils import *

########################################################################
# Gated Transformer Train
def gated_model_train():
    """Runs preprocessing code, initializes parameters, generates models, optimizers, criterion,
    and run the trainining function
    """
    # Initialize iterator, SRC field, and TRG field
    (
        train_iterator,
        valid_iterator,
        test_iterator,
        SRC,
        TRG,
    ) = run_preprocess()  # this can't be subscripted

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

    # Initializes Encoder layers

    # Initializes Decoder layers

    # Initializes other helps layers....

    # Convert tokenized string into integers
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    # Initializes the model's weights

    # Initializes Adam Optimizer

    # Initializes Cross Entropy Loss Function

    # Variables for printing format
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

    return


if __name__ == "__main__":
    gated_model_train()

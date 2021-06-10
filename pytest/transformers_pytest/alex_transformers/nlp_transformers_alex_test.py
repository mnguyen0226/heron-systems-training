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
    HID_DIM = 256 # this is the features
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

    # Initialize the Embedding layer
    emb = Embedding(input_dim=INPUT_DIM, hid_dim=HID_DIM, dropout=ENC_DROPOUT)

    # Initializes Encoder layer
    gated_encoder = Encoder(in_shape=[])

    # Initializes Decoder layer
    gated_decoder = 0

    # Convert tokenized string into integers
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    # Initializes the seq2seq model
    model = Seq2Seq(
        embedding=emb,
        encoder=gated_encoder,
        decoder=gated_decoder,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
    )    

    # Initializes the model's weights
    model.apply(initialize_weights)

    # Initializes Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    # Initializes Cross Entropy Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

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
    (
        gated_transformers_training_loss,
        gated_transformers_validating_loss,
        gated_transformers_training_PPL,
        gated_transformers_validating_PPL,
    ) = gated_transformers_main(
        model=model,
        train_iterator=train_iterator,
        optimizer=optimizer,
        criterion=criterion,
        CLIP=CLIP,
        valid_iterator=valid_iterator,
        n_epochs=N_EPOCHS,
    )

    return (
        gated_transformers_training_loss,
        gated_transformers_validating_loss,
        gated_transformers_training_PPL,
        gated_transformers_validating_PPL,
    )


# Run Training Loop:
# Run training Gated Transformers
(
    gated_transformers_training_loss,
    gated_transformers_validating_loss,
    gated_transformers_training_PPL,
    gated_transformers_validating_PPL,
) = gated_model_train()

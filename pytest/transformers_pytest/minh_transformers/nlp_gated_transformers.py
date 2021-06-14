from decimal import Decimal
import torch
import torch.nn as nn
from utils.preprocess import run_preprocess, device

from utils.alex_gated_transformers.gated_training_utils import ( 
    gated_transformers_main,
    test_gated_transformers_model,
    initialize_weights,
)
from utils.alex_gated_transformers.gated_encoder import EncoderLayers 
from utils.alex_gated_transformers.gated_decoder import DecoderLayers
from utils.alex_gated_transformers.gated_seq2seq import (
    Seq2Seq,
    EmbeddingEncLayer,
    EmbeddingDecLayer,
)

from utils.original_transformers.training_utils import {
    origin_transformers_main, test_origin_transformers_model, initialize_weights

}

##################################################################################################
# GLOBAL VARIABLE FOR TESTING
# initializer training iterator, SRC field, TRG field
(
    train_iterator,
    valid_iterator,
    test_iterator,
    SRC,
    TRG,
) = run_preprocess()

# Initialize variables for Gated Transformers. This can be adjusted
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
GATED_ENC_LAYERS = 1 # 3 
GATED_DEC_LAYERS = 1
GATED_ENC_HEADS = 8
GATED_DEC_HEADS = 8
ENC_PF_DIM = 256  # 512
DEC_PF_DIM = 256
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

# Initializes variables for training process. This can be adjusted
N_EPOCHS = 10
CLIP = 1
LEARNING_RATE = 0.0005

# Defines whole Seq2Seq encapsulating model. Convert tokenized string to integers
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]


##################################################################################################
# ALEX GATED TRANSFORMERS
def alex_gated_model_train():

    emb_enc = EmbeddingEncLayer(
        input_dim=INPUT_DIM, hid_dim=HID_DIM, dropout=ENC_DROPOUT
    )
    emb_dec = EmbeddingDecLayer(
        output_dim=OUTPUT_DIM, hid_dim=HID_DIM, dropout=DEC_DROPOUT
    )

    enc = EncoderLayers(
        in_shape=[HID_DIM, 1],
        n_layers=GATED_ENC_LAYERS,
        nb_heads=GATED_ENC_HEADS,
        dropout=ENC_DROPOUT,
    )

    dec = DecoderLayers(
        output_dim=OUTPUT_DIM,
        encoder_output_shape=[HID_DIM, 1],
        n_layers=GATED_DEC_LAYERS,
        nb_heads=GATED_DEC_HEADS,
        dropout=DEC_DROPOUT,
    )

    model = Seq2Seq(
        embedding_enc=emb_enc,
        embedding_dec=emb_dec,
        encoder=enc,
        decoder=dec,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
    ).to(device)

    # Initializes the model's weights - Xavier
    model.apply(initialize_weights)

    # Initializes Adam optimizer for updating weight
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initializes Cross Entropy Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # Variables for printing format
    gated_transformers_enc_layers = GATED_ENC_LAYERS
    gated_transformers_dec_layers = GATED_DEC_LAYERS
    gated_transformers_enc_nb_heads = GATED_ENC_HEADS
    gated_transformers_dec_nb_heads = GATED_DEC_HEADS

    # results from Gated Transformers
    print(
        "\n------------------------------------------------------------------------------------------"
    )
    print(
        f"\nThe gated Transformer has {gated_transformers_enc_nb_heads} encoder head(s), {gated_transformers_dec_nb_heads} \
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

    print("----------------------------------------")
    print("Traininng Gated Transformers Testing Set")
    print("----------------------------------------")
    (
        gated_transformers_testing_loss,
        gated_transformers_testing_PPL,
    ) = test_gated_transformers_model(
        model=model, test_iterator=test_iterator, criterion=criterion
    )

    return (
        gated_transformers_training_loss,
        gated_transformers_validating_loss,
        gated_transformers_training_PPL,
        gated_transformers_validating_PPL,
        gated_transformers_testing_loss,
        gated_transformers_testing_PPL,
    )

##################################################################################################
# Run training Alex Gated Transformers
(
    gated_transformers_training_loss,
    gated_transformers_validating_loss,
    gated_transformers_training_PPL,
    gated_transformers_validating_PPL,
    gated_transformers_testing_loss,
    gated_transformers_testing_PPL,
) = alex_gated_model_train()

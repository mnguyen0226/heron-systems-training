# The goal is to create a training loop working
# fix the encoder split similar to Alex
# fix the decoder split similar to Alex

from decimal import Decimal
import torch
import torch.nn as nn
from utils.preprocess import run_preprocess, device

from utils.gated_transformers.training_utils import (  # training, testing loop
    gated_transformers_main,
    test_gated_transformers_model,
    initialize_weights,
)
from utils.gated_transformers.encoder import Encoder  # encoder
from utils.gated_transformers.decoder import Decoder  # decoder
from utils.gated_transformers.seq2seq import Seq2Seq  # seqseq

def gated_model_train():

    # initializer training iterator, SRC field, TRG field
    (train_iterator, valid_iterator, test_iterator, SRC, TRG,) = run_preprocess()

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

    # Defines whole Seq2Seq encapsulating model. Convert tokenized string to integers
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    enc = Encoder(
        input_dim=INPUT_DIM,
        hid_dim=HID_DIM,
        n_layers=GATED_ENC_LAYERS,
        n_heads=GATED_ENC_HEADS,
        pf_dim=ENC_PF_DIM,
        dropout=ENC_DROPOUT,
        device=device,
    )

    dec = Decoder(
        output_dim=OUTPUT_DIM,
        hid_dim=HID_DIM,
        n_layers=GATED_DEC_LAYERS,
        n_heads=GATED_DEC_HEADS,
        pf_dim=DEC_PF_DIM,
        dropout=DEC_DROPOUT,
        device=device,
    )

    model = Seq2Seq(
        encoder=enc,
        decoder=dec,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
        device=device,
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

# Run training Gated Transformers
(
    gated_transformers_training_loss,
    gated_transformers_validating_loss,
    gated_transformers_training_PPL,
    gated_transformers_validating_PPL,
    gated_transformers_testing_loss,
    gated_transformers_testing_PPL,
) = gated_model_train()

# pytest script that make sure the final train/validate loss and train/validate PPL of the gated transformers is
# better than the original transformers from "Attention Is All You Need"
# We want the Train Loss, Val Loss, Train PPL, and Val PPL to be lower than the test bench
# RUNNING COMMAND: pytest -s pytest_transformers.py

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

from utils.gated_transformers.training_utils import (
    gated_transformers_main,
    test_gated_transformers_model,
)
from utils.gated_transformers.encoder import Encoder
from utils.gated_transformers.decoder import Decoder
from utils.gated_transformers.seq2seq import Seq2Seq
from utils.gated_transformers.training_utils import (
    initialize_weights,
    gated_transformers_main,
    test_gated_transformers_model,
)

########################################################################
# Gated Transformers Train
def gated_model_train() -> Tuple[float, float, float, float, float, float]:
    """Creates a training function for Gated Transformers

    Return
    ----------
    gated_transformers_training_loss: float
        Gated transformers training loss
    gated_transformers_validating_loss: float
        Gated transformers validating loss
    gated_transformers_training_PPL: float
        Gated Transformers training PPL
    gated_transformers_validating_PPL: float
        Gated Transformers validating PPL
    gated_transformers_testing_loss: float
        Gated Transformers testing loss
    gated_transformers_testing_PPL: float
        Gated Transformers testing PPL
    """
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

    # Initializes Encoder layers
    enc = Encoder(
        INPUT_DIM, HID_DIM, GATED_ENC_LAYERS, GATED_ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device
    )

    # Initializes Decoder layers
    dec = Decoder(
        OUTPUT_DIM, HID_DIM, GATED_DEC_LAYERS, GATED_DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device
    )

    # Defines whole Seq2Seq encapsulating model
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    # Initializes model
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    # Initializes the model's weights
    model.apply(initialize_weights)

    # Initializes Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initializes Cross Entropy Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

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
    ) = test_gated_transformers_model(model=model, test_iterator=test_iterator, criterion=criterion)

    return (
        gated_transformers_training_loss,
        gated_transformers_validating_loss,
        gated_transformers_training_PPL,
        gated_transformers_validating_PPL,
        gated_transformers_testing_loss,
        gated_transformers_testing_PPL,
    )


########################################################################
# Original Transformers Train
def original_model_train() -> Tuple[float, float, float, float, float, float]:
    """Creates a training function for Original Transformers

    Return
    ----------
    original_transformers_training_loss: float
        Original transformers training loss
    original_transformers_validating_loss: float
        Original transformers validating loss
    original_transformers_training_PPL: float
        Original Transformers training PPL
    original_transformers_validating_PPL: float
        Original Transformers validating PPL
    original_transformers_testing_loss: float
        Original Transformers testing loss
    original_transformers_testing_PPL: float
        Original Transformers testing PPL
    """
    # Initialize iterator, SRC field, and TRG field
    train_iterator, valid_iterator, test_iterator, SRC, TRG = run_preprocess()

    # Initialize variables for Gated Transformers. This can be adjusted
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    # Initializes variables for training process. This can be adjusted
    N_EPOCHS = 10
    CLIP = 1
    LEARNING_RATE = 0.0005

    # Initializes Encoder layers
    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)

    # Initializes Decoder layers
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

    # Define whole Seq2Seq encapsulating model
    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    # Initializes model
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    # Initialize the model's weights
    model.apply(initialize_weights)

    # Initializes Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initializes Cross Entropy Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # Variables for printint format
    origin_transformers_enc_layers = ENC_LAYERS
    origin_transformers_dec_layers = DEC_LAYERS
    origin_transformers_enc_n_heads = ENC_HEADS
    origin_transformers_dec_n_heads = DEC_HEADS

    # results from Transformers "Attention Is All You Need"
    print(
        f"\nThe original Transformer has {origin_transformers_enc_n_heads} encoder head(s), {origin_transformers_dec_n_heads} \
    decoder head(s), {origin_transformers_enc_layers} encoder layer(s), {origin_transformers_dec_layers} decoder layer(s)"
    )

    print("---------------------------------------------")
    print("Training Test Bench Transformers Training Set")
    print("---------------------------------------------")
    (
        origin_transformers_training_loss,
        origin_transformers_validating_loss,
        origin_transformers_training_PPL,
        origin_transformers_validating_PPL,
    ) = origin_transformers_main(
        model=model,
        train_iterator=train_iterator,
        optimizer=optimizer,
        criterion=criterion,
        CLIP=CLIP,
        valid_iterator=valid_iterator,
        n_epochs=N_EPOCHS,
    )

    print("\n--------------------------------------------")
    print("Training Test Bench Transformers Testing Set")
    print("--------------------------------------------")
    (
        origin_transformers_testing_loss,
        origin_transformers_testing_PPL,
    ) = test_origin_transformers_model(
        model=model, test_iterator=test_iterator, criterion=criterion
    )

    return (
        origin_transformers_training_loss,
        origin_transformers_validating_loss,
        origin_transformers_training_PPL,
        origin_transformers_validating_PPL,
        origin_transformers_testing_loss,
        origin_transformers_testing_PPL,
    )


########################################################################
# Test
# Run training Gated Transformers
(
    gated_transformers_training_loss,
    gated_transformers_validating_loss,
    gated_transformers_training_PPL,
    gated_transformers_validating_PPL,
    gated_transformers_testing_loss,
    gated_transformers_testing_PPL,
) = gated_model_train()

# Run training Original Transformers
(
    origin_transformers_training_loss,
    origin_transformers_validating_loss,
    origin_transformers_training_PPL,
    origin_transformers_validating_PPL,
    origin_transformers_testing_loss,
    origin_transformers_testing_PPL,
) = original_model_train()


class TestGatedTransformersTrainingSet:
    def test_training_loss(self):
        """Validated the training loss of gated transformers < original transformers'"""
        global gated_transformers_training_loss, origin_transformers_training_loss
        assert Decimal(gated_transformers_training_loss) < Decimal(
            origin_transformers_training_loss
        )

    def test_validating_loss(self):
        """Validates the validating loss of gated transformers < original transformers'"""
        global gated_transformers_validating_loss, origin_transformers_validating_loss
        assert Decimal(gated_transformers_validating_loss) < Decimal(
            origin_transformers_validating_loss
        )

    def test_training_PPL(self):
        """Validates the training PPL of gated transformers < original transformers'"""
        global gated_transformers_training_PPL, origin_transformers_training_PPL
        assert Decimal(gated_transformers_training_PPL) < Decimal(origin_transformers_training_PPL)

    def test_validating_PPL(self):
        """Validates the validating PPL of gated transfomers < original transformers'"""
        global gated_transformers_validating_PPL, origin_transformers_validating_PPL
        assert Decimal(gated_transformers_validating_PPL) < Decimal(
            origin_transformers_validating_PPL
        )


class TestGatedTransformersTestingSet:
    def test_testing_loss(self):
        """Validates the testing loss of the gated transformers < origin transformers'"""
        global gated_transformers_testing_loss, origin_transformers_testing_loss
        assert Decimal(gated_transformers_testing_loss) < Decimal(origin_transformers_testing_loss)

    def test_testing_PPL(self):
        """Validates the testing PPL of the gated transformers < origin transformers'"""
        global gated_transformers_testing_PPL, origin_transformers_testing_PPL
        assert Decimal(gated_transformers_testing_PPL) < Decimal(origin_transformers_testing_PPL)

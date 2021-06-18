from decimal import Decimal
import torch
import torch.nn as nn
from utils.preprocess import run_preprocess, device
from typing import Tuple

# Import methods for Alex Gated Transformers
from utils.gated_transformers.gated_training_utils import (
    gated_transformers_main,
    validate_gated_transformers_model,
    gated_initialize_weights,
)
from utils.gated_transformers.gated_encoder_layers import EncoderLayers
from utils.gated_transformers.gated_decoder_layers import DecoderLayers
from utils.gated_transformers.gated_seq2seq import (
    GatedSeq2Seq,
    EmbeddingEncLayer,
    EmbeddingDecLayer,
)

# Import methods for Original Gated Transformers
from utils.original_transformers.original_training_utils import (
    origin_transformers_main,
    validate_origin_transformers_model,
    origin_initialize_weights,
)
from utils.original_transformers.original_encoder import Encoder
from utils.original_transformers.original_decoder import Decoder
from utils.original_transformers.original_seq2seq import Seq2Seq

# Import methods for Masked Gated Transformers
from utils.masked_gated_transformers.masked_gated_training_utils import (
    masked_gated_transformers_main,
    masked_initialize_weights,
    validate_masked_gated_transformers_model,
)
from utils.masked_gated_transformers.masked_gated_encoder import MaskedEncoder
from utils.masked_gated_transformers.masked_gated_decoder import MaskedDecoder
from utils.masked_gated_transformers.masked_gated_seq2seq import MaskedSeq2Seq

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
ENC_LAYERS = 1  # 3
DEC_LAYERS = 1
ENC_HEADS = 8
DEC_HEADS = 8
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
def alex_gated_model_train() -> Tuple[float, float, float, float, float, float]:
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
    emb_enc = EmbeddingEncLayer(
        input_dim=INPUT_DIM, hid_dim=HID_DIM, dropout=ENC_DROPOUT
    )
    emb_dec = EmbeddingDecLayer(
        output_dim=OUTPUT_DIM, hid_dim=HID_DIM, dropout=DEC_DROPOUT
    )

    enc = EncoderLayers(
        in_shape=[HID_DIM, 1],
        n_layers=ENC_LAYERS,
        nb_heads=ENC_HEADS,
        dropout=ENC_DROPOUT,
    )

    dec = DecoderLayers(
        output_dim=OUTPUT_DIM,
        encoder_output_shape=[HID_DIM, 1],
        n_layers=DEC_LAYERS,
        nb_heads=DEC_HEADS,
        dropout=DEC_DROPOUT,
    )

    model = GatedSeq2Seq(
        embedding_enc=emb_enc,
        embedding_dec=emb_dec,
        encoder=enc,
        decoder=dec,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
    ).to(device)

    # Initializes the model's weights - Xavier
    model.apply(gated_initialize_weights)

    # Initializes Adam optimizer for updating weight
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initializes Cross Entropy Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # Variables for printing format
    gated_transformers_enc_layers = ENC_LAYERS
    gated_transformers_dec_layers = DEC_LAYERS
    gated_transformers_enc_nb_heads = ENC_HEADS
    gated_transformers_dec_nb_heads = DEC_HEADS

    # results from Gated Transformers
    print(
        "\n-----------------------------------------------------------------------------------------------------------------"
    )
    print(
        f"\nThe Gated Transformer has {gated_transformers_enc_nb_heads} encoder head(s), {gated_transformers_dec_nb_heads}\
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
    ) = validate_gated_transformers_model(
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
# ORIGINAL "ATTENTION IS ALL YOU NEED" TRANSFORMERS
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
    # Initializes Encoder layers
    enc = Encoder(
        input_dim=INPUT_DIM,
        hid_dim=HID_DIM,
        n_layers=ENC_LAYERS,
        n_heads=ENC_HEADS,
        pf_dim=ENC_PF_DIM,
        dropout=ENC_DROPOUT,
        device=device,
    )

    # Initializes Decoder layers
    dec = Decoder(
        output_dim=OUTPUT_DIM,
        hid_dim=HID_DIM,
        n_layers=DEC_LAYERS,
        n_heads=DEC_HEADS,
        pf_dim=DEC_PF_DIM,
        dropout=DEC_DROPOUT,
        device=device,
    )

    # Initializes model
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    # Initialize the model's weights
    model.apply(origin_initialize_weights)

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
        "\n-----------------------------------------------------------------------------------------------------------------"
    )
    print(
        f"\nThe Original Transformer has {origin_transformers_enc_n_heads} encoder head(s), {origin_transformers_dec_n_heads}\
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
    ) = validate_origin_transformers_model(
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


##################################################################################################
# MASKED GATED TRANSFORMERS
def masked_model_train() -> Tuple[float, float, float, float, float, float]:
    """Creates a training function for Masked Gated Transformers

    Return
    ----------
    masked_transformers_training_loss: float
        Masked gated transformers training loss
    masked_transformers_validating_loss: float
        Masked gated transformers validating loss
    masked_transformers_training_PPL: float
        Masked gated Transformers training PPL
    masked_transformers_validating_PPL: float
        Masked gated Transformers validating PPL
    masked_transformers_testing_loss: float
        Masked gated Transformers testing loss
    masked_transformers_testing_PPL: float
        Masked gated Transformers testing PPL
    """
    # Initializes Encoder layers, not putting input sentence
    enc = MaskedEncoder(
        input_dim=INPUT_DIM,
        hid_dim=HID_DIM,
        n_layers=ENC_LAYERS,
        n_heads=ENC_HEADS,
        pf_dim=ENC_PF_DIM,
        dropout=ENC_DROPOUT,
        device=device,
    )

    # Initializes Decoder layers
    dec = MaskedDecoder(
        output_dim=OUTPUT_DIM,
        hid_dim=HID_DIM,
        n_layers=DEC_LAYERS,
        n_heads=DEC_HEADS,
        pf_dim=DEC_PF_DIM,
        dropout=DEC_DROPOUT,
        device=device,
    )

    # Initializes model
    model = MaskedSeq2Seq(
        encoder=enc,
        decoder=dec,
        src_pad_idx=SRC_PAD_IDX,
        trg_pad_idx=TRG_PAD_IDX,
        device=device,
    ).to(device)

    # Initializes the model's weights
    model.apply(masked_initialize_weights)

    # Initializes Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initializes Cross Entropy Loss Function
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    # Variables for printing format
    masked_transformers_enc_layers = ENC_LAYERS
    masked_transformers_dec_layers = DEC_LAYERS
    masked_transformers_enc_n_heads = ENC_HEADS
    masked_transformers_dec_n_heads = DEC_HEADS

    # results from Masked Gated Transformers
    print(
        "\n-----------------------------------------------------------------------------------------------------------------"
    )
    print(
        f"\nThe Masked Gated Transformer has {masked_transformers_enc_n_heads} encoder head(s), {masked_transformers_dec_n_heads} \
    decoder head(s), {masked_transformers_enc_layers} encoder layer(s), {masked_transformers_dec_layers} decoder layer(s)"
    )

    print("------------------------------------------------")
    print("Traininng Masked Gated Transformers Training Set")
    print("------------------------------------------------")
    (
        masked_transformers_training_loss,
        masked_transformers_validating_loss,
        masked_transformers_training_PPL,
        masked_transformers_validating_PPL,
    ) = masked_gated_transformers_main(
        model=model,
        train_iterator=train_iterator,
        optimizer=optimizer,
        criterion=criterion,
        CLIP=CLIP,
        valid_iterator=valid_iterator,
        n_epochs=N_EPOCHS,
    )

    print("-----------------------------------------------")
    print("Traininng Masked Gated Transformers Testing Set")
    print("-----------------------------------------------")
    (
        masked_transformers_testing_loss,
        masked_transformers_testing_PPL,
    ) = validate_masked_gated_transformers_model(
        model=model, test_iterator=test_iterator, criterion=criterion
    )

    return (
        masked_transformers_training_loss,
        masked_transformers_validating_loss,
        masked_transformers_training_PPL,
        masked_transformers_validating_PPL,
        masked_transformers_testing_loss,
        masked_transformers_testing_PPL,
    )


##################################################################################################
# Run training Original Transformers
(
    origin_transformers_training_loss,
    origin_transformers_validating_loss,
    origin_transformers_training_PPL,
    origin_transformers_validating_PPL,
    origin_transformers_testing_loss,
    origin_transformers_testing_PPL,
) = original_model_train()

# Run training Alex Gated Transformers
(
    gated_transformers_training_loss,
    gated_transformers_validating_loss,
    gated_transformers_training_PPL,
    gated_transformers_validating_PPL,
    gated_transformers_testing_loss,
    gated_transformers_testing_PPL,
) = alex_gated_model_train()

# Run training Masked Gated Transformers
(
    masked_transformers_training_loss,
    masked_transformers_validating_loss,
    masked_transformers_training_PPL,
    masked_transformers_validating_PPL,
    masked_transformers_testing_loss,
    masked_transformers_testing_PPL,
) = masked_model_train()


##################################################################################################
# PYTEST ALEX GATED TRANSFORMERS VS ORIGINAL TRANSFORMERS
class TestGatedTransformersValidatingSet1:
    def test_validating_loss_1(self):
        """Validates the validating loss of gated transformers < original transformers'"""
        global gated_transformers_validating_loss, origin_transformers_validating_loss
        assert Decimal(gated_transformers_validating_loss) < Decimal(
            origin_transformers_validating_loss
        )

    def test_validating_PPL_1(self):
        """Validates the validating PPL of gated transfomers < original transformers'"""
        global gated_transformers_validating_PPL, origin_transformers_validating_PPL
        assert Decimal(gated_transformers_validating_PPL) < Decimal(
            origin_transformers_validating_PPL
        )


class TestGatedTransformersTestingSet1:
    def test_testing_loss_1(self):
        """Validates the testing loss of the gated transformers < origin transformers'"""
        global gated_transformers_testing_loss, origin_transformers_testing_loss
        assert Decimal(gated_transformers_testing_loss) < Decimal(
            origin_transformers_testing_loss
        )

    def test_testing_PPL_1(self):
        """Validates the testing PPL of the gated transformers < origin transformers'"""
        global gated_transformers_testing_PPL, origin_transformers_testing_PPL
        assert Decimal(gated_transformers_testing_PPL) < Decimal(
            origin_transformers_testing_PPL
        )


# PYTEST ALEX GATED TRANSFORMERS VS MASKED TRANSFORMERS (MASKED TRANSFORMER IS EXPECTED TO PERFORMED BEST)
class TestGatedTransformersValidatingSet2:
    def test_validating_loss_2(_self):
        """Validates the validating loss of gated transformers > masked gated transformers'"""
        global gated_transformers_validating_loss, masked_transformers_validating_loss
        assert Decimal(gated_transformers_validating_loss) > Decimal(
            masked_transformers_validating_loss
        )

    def test_validating_PPL_2(self):
        """Validates the validating PPL of gated transfomers > masked gated transformers'"""
        global gated_transformers_validating_PPL, masked_transformers_validating_PPL
        assert Decimal(gated_transformers_validating_PPL) > Decimal(
            masked_transformers_validating_PPL
        )


class TestGatedTransformersTestingSet2:
    def test_testing_loss_2(self):
        """Validates the testing loss of the gated transformers > masked gated transformers'"""
        global gated_transformers_testing_loss, masked_transformers_testing_loss
        assert Decimal(gated_transformers_testing_loss) > Decimal(
            masked_transformers_testing_loss
        )

    def test_testing_PPL_2(self):
        """Validates the testing PPL of the gated transformers > masked gated transformers'"""
        global gated_transformers_testing_PPL, masked_transformers_testing_PPL
        assert Decimal(gated_transformers_testing_PPL) > Decimal(
            masked_transformers_testing_PPL
        )

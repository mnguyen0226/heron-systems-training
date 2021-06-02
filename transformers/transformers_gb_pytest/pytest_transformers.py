# pytest script that make sure the final train/validate loss and train/validate PPL of the gated transformers is better than the original transformers from "Attention Is All You Need"
# We want the Train Loss, Val Loss, Train PPL, and Val PPL to be lower than the test bench

import pytest
from gated_transformers.main import *
from original_transformers.main import *

# results from Transformers "Attention Is All You Need"
origin_transformers_training_loss = 0.0
origin_transformers_validating_loss = 0.0
origin_transformers_training_PPL = 0.0
origin_transformers_validating_PPL = 0.0

# results from Gated Transformers 
gated_transformers_training_loss = 0.0
gated_transformers_validating_loss = 0.0
gated_transformers_training_PPL = 0.0
gated_transformers_validating_PPL = 0.0

# information for printing wise
origin_transformers_enc_n_heads = ENC_HEADS
origin_transformers_dec_n_heads = DEC_HEADS
origin_transformers_enc_layers = ENC_LAYERS
origin_transformers_dec_layers = DEC_LAYERS

gated_transformers_enc_n_heads = GATED_ENC_HEADS
gated_transformers_dec_n_heads = GATED_DEC_HEADS
gated_transformers_enc_layers = GATED_ENC_LAYERS
gated_transformers_dec_layers = GATED_DEC_LAYERS

def run_original_transformers():
    # print(origin_transformers_enc_n_heads)
    # print(origin_transformers_dec_n_heads)
    # print(origin_transformers_enc_layers)
    # print(origin_transformers_dec_layers)

    origin_transformers_training_loss, origin_transformers_validating_loss, origin_transformers_training_PPL, origin_transformers_validating_PPL = origin_transformers_main()
    print(origin_transformers_training_loss)
    print(origin_transformers_validating_loss)
    print(origin_transformers_training_PPL)
    print(origin_transformers_validating_PPL)

def run_gated_transformers():
    # print(gated_transformers_enc_n_heads)
    # print(gated_transformers_dec_n_heads)
    # print(gated_transformers_enc_layers)
    # print(gated_transformers_dec_layers)

    gated_transformers_training_loss, gated_transformers_validating_loss, gated_transformers_training_PPL, gated_transformers_validating_PPL = gated_transformers_main()
    print(gated_transformers_training_loss)
    print(gated_transformers_validating_loss)
    print(gated_transformers_training_PPL)
    print(gated_transformers_training_PPL)

# class TestGatedTransformers:
#     def test
# run the two function above then pytest

if __name__ == "__main__":
    run_gated_transformers()
    run_original_transformers()
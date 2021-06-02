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
    origin_transformers_training_loss, origin_transformers_validating_loss, origin_transformers_training_PPL, origin_transformers_validating_PPL = origin_transformers_main()

    # for testing
    print(origin_transformers_training_loss)
    print(origin_transformers_validating_loss)
    print(origin_transformers_training_PPL)
    print(origin_transformers_validating_PPL)

def run_gated_transformers():
    gated_transformers_training_loss, gated_transformers_validating_loss, gated_transformers_training_PPL, gated_transformers_validating_PPL = gated_transformers_main()

    # for testing
    print(gated_transformers_training_loss)
    print(gated_transformers_validating_loss)
    print(gated_transformers_training_PPL)
    print(gated_transformers_training_PPL)

# run main() of the two transformers
run_gated_transformers()
run_original_transformers()

print(f"The original Transformer has {origin_transformers_enc_n_heads} encoder head(s), {origin_transformers_dec_n_heads} decoder head(s), {origin_transformers_enc_layers} encoder layer(s), {origin_transformers_dec_layers} decoder layer(s)") 
print(f"The gated Transformer has {gated_transformers_enc_n_heads} encoder head(s), {gated_transformers_dec_n_heads} decoder head(s), {gated_transformers_enc_layers} encoder layer(s), {gated_transformers_dec_layers} decoder layer(s)")

class TestGatedTransformers:
    def test_training_loss(self):
        """Validated the training loss of gated transformers < original transformers'
        """
        assert gated_transformers_training_loss < origin_transformers_training_loss

    def test_validating_loss(self):
        """Validates the testing loss of gated transformers < original transformers'
        """
        assert gated_transformers_validating_loss < origin_transformers_validating_loss

    def test_training_PPL(self):
        """Validates the training PPL of gated transformers < original transformers'
        """
        assert gated_transformers_training_PPL < origin_transformers_training_PPL
    
    def test_validating_PPL(self):
        """Validates the testing PPL of gated transfomers < original transformers' 
        """
        assert gated_transformers_validating_PPL < origin_transformers_validating_PPL
# pytest script that make sure the final train/validate loss and train/validate PPL of the gated transformers is better than the original transformers from "Attention Is All You Need"
# We want the Train Loss, Val Loss, Train PPL, and Val PPL to be lower than the test bench
# RUNNING COMMAND: pytest -s pytest_transformers.py

import pytest
from gated_transformers.main import *
from original_transformers.main import *
from decimal import Decimal

# information for printing wise
origin_transformers_enc_n_heads = ENC_HEADS
origin_transformers_dec_n_heads = DEC_HEADS
origin_transformers_enc_layers = ENC_LAYERS
origin_transformers_dec_layers = DEC_LAYERS

gated_transformers_enc_n_heads = GATED_ENC_HEADS
gated_transformers_dec_n_heads = GATED_DEC_HEADS
gated_transformers_enc_layers = GATED_ENC_LAYERS
gated_transformers_dec_layers = GATED_DEC_LAYERS

# results from Transformers "Attention Is All You Need"
print(f"\nThe original Transformer has {origin_transformers_enc_n_heads} encoder head(s), {origin_transformers_dec_n_heads} decoder head(s), {origin_transformers_enc_layers} encoder layer(s), {origin_transformers_dec_layers} decoder layer(s)") 
print("------------------------------")
print("Training Test Bench Transformers")
origin_transformers_training_loss, origin_transformers_validating_loss, origin_transformers_training_PPL, origin_transformers_validating_PPL = origin_transformers_main()

# results from Gated Transformers 
print(f"\nThe gated Transformer has {gated_transformers_enc_n_heads} encoder head(s), {gated_transformers_dec_n_heads} decoder head(s), {gated_transformers_enc_layers} encoder layer(s), {gated_transformers_dec_layers} decoder layer(s)")
print("------------------------------")
print("Traininng Gated Transformers")
gated_transformers_training_loss, gated_transformers_validating_loss, gated_transformers_training_PPL, gated_transformers_validating_PPL = gated_transformers_main()

class TestGatedTransformersTrainingSet:
    def test_training_loss(self):
        """Validated the training loss of gated transformers < original transformers'
        """
        global gated_transformers_training_loss, origin_transformers_training_loss
        assert Decimal(gated_transformers_training_loss) < Decimal(origin_transformers_training_loss)

    def test_validating_loss(self):
        """Validates the testing loss of gated transformers < original transformers'
        """
        global gated_transformers_validating_loss, origin_transformers_validating_loss
        assert Decimal(gated_transformers_validating_loss) <  Decimal(origin_transformers_validating_loss)

    def test_training_PPL(self):
        """Validates the training PPL of gated transformers < original transformers'
        """
        global gated_transformers_training_PPL, origin_transformers_training_PPL
        assert Decimal(gated_transformers_training_PPL) < Decimal(origin_transformers_training_PPL)
    
    def test_validating_PPL(self):
        """Validates the testing PPL of gated transfomers < original transformers' 
        """
        global gated_transformers_validating_PPL, origin_transformers_validating_PPL
        assert Decimal(gated_transformers_validating_PPL) < Decimal(origin_transformers_validating_PPL)


class TestGatedTransformersTestingSet:
    def test_testing_loss(self):
        assert Decimal()
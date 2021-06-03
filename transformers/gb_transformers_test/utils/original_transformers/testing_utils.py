# script test trained original transformers model

from utils.original_transformers.training_utils import *

def test_origin_transformers_model():
    model.load_state_dict(torch.load('original-tut6-model.pt'))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    return test_loss, math.exp(test_loss)

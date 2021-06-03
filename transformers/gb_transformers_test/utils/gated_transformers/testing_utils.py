# script testing trained gated transformers model

from utils.gated_transformers.training_utils import *

def test_gated_transformers_model():
    """Testing trained Gated Transformer
    """
    model.load_state_dict(torch.load('gated-tut6-model.pt', map_location=torch.device('cpu')))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    return test_loss, math.exp(test_loss)

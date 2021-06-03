from original_transformers.main import *

def test_model():
    model.load_state_dict(torch.load('tut6-model.pt'))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    return test_loss, math.exp(test_loss)

if __name__ == "__main__":
    test_model()
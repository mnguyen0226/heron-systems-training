from main import *


def test_model():
    """Testing trained Gated Transformer"""
    model.load_state_dict(torch.load("tut6-model.pt", map_location=torch.device("cpu")))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")


if __name__ == "__main__":
    test_model()

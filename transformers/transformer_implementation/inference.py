# Resource: https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb
from main import *
from utils.preprocess import *
from utils.model import *

"""
Inference

Now we can can translations from our model with the translate_sentence function below.

The steps taken are:

    tokenize the source sentence if it has not been tokenized (is a string)
    append the <sos> and <eos> tokens
    numericalize the source sentence
    convert it to a tensor and add a batch dimension
    create the source sentence mask
    feed the source sentence and mask into the encoder
    create a list to hold the output sentence, initialized with an <sos> token
    while we have not hit a maximum length
        convert the current output sentence prediction into a tensor with a batch dimension
        create a target sentence mask
        place the current output, encoder output and both masks into the decoder
        get next output token prediction from decoder along with attention
        add prediction to current output sentence prediction
        break if the prediction was an <eos> token
    convert the output sentence from indexes to tokens
    return the output sentence (with the <sos> token removed) and the attention from the last layer
"""


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):

    model.eval()

    if isinstance(sentence, str):
        nlp = spacy.load("de_core_news_sm")
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention


def display_attention(sentence, translation, attention, n_heads=8, n_rows=4, n_cols=2):

    assert n_rows * n_cols == n_heads

    fig = plt.figure(figsize=(15, 25))

    for i in range(n_heads):

        ax = fig.add_subplot(n_rows, n_cols, i + 1)

        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap="bone")

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(
            [""] + ["<sos>"] + [t.lower() for t in sentence] + ["<eos>"], rotation=45
        )
        ax.set_yticklabels([""] + translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def test_training_set():
    example_idx = 8

    src = vars(train_data.examples[example_idx])["src"]
    trg = vars(train_data.examples[example_idx])["trg"]

    print(f"src = {src}")
    print(f"trg = {trg}")

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f"predicted trg = {translation}")

    display_attention(src, translation, attention)


def test_validating_set():

    example_idx = 6

    src = vars(valid_data.examples[example_idx])["src"]
    trg = vars(valid_data.examples[example_idx])["trg"]

    print(f"src = {src}")
    print(f"trg = {trg}")

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f"predicted trg = {translation}")

    display_attention(src, translation, attention)


def test_testing_set():
    example_idx = 10

    src = vars(test_data.examples[example_idx])["src"]
    trg = vars(test_data.examples[example_idx])["trg"]

    print(f"src = {src}")
    print(f"trg = {trg}")

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f"predicted trg = {translation}")
    display_attention(src, translation, attention)


def run_test():
    test_training_set()
    test_validating_set()
    test_testing_set()

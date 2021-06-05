from torchtext.data.metrics import bleu_score
from main import *
from inference import *


def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):

    trgs = []
    pred_trgs = []

    for datum in data:

        src = vars(datum)["src"]
        trg = vars(datum)["trg"]

        pred_trg, _ = translate_sentence(
            src, src_field, trg_field, model, device, max_len
        )

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)


def run_test():
    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

    print(f"BLEU score = {bleu_score*100:.2f}")


########################################################################
# the calculate_bleu function is unoptimized, this is a faster version
def translate_sentence_vectorized(
    src_tensor, src_field, trg_field, model, device, max_len=50
):
    assert isinstance(src_tensor, torch.Tensor)

    model.eval()
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    # enc_src = [batch_sz, src_len, hid_dim]

    trg_indexes = [
        [trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))
    ]
    # Even though some examples might have been completed by producing a <eos> token
    # we still need to feed them through the model because other are not yet finished
    # and all examples act as a batch. Once every single sentence prediction encounters
    # <eos> token, then we can stop predicting.
    translations_done = [0] * len(src_tensor)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_tokens = output.argmax(2)[:, -1]
        for i, pred_token_i in enumerate(pred_tokens):
            trg_indexes[i].append(pred_token_i)
            if pred_token_i == trg_field.vocab.stoi[trg_field.eos_token]:
                translations_done[i] = 1
        if all(translations_done):
            break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
        pred_sentence = []
        for i in range(1, len(trg_sentence)):
            if trg_sentence[i] == trg_field.vocab.stoi[trg_field.eos_token]:
                break
            pred_sentence.append(trg_field.vocab.itos[trg_sentence[i]])
        pred_sentences.append(pred_sentence)

    return pred_sentences, attention


def calculate_bleu_alt(iterator, src_field, trg_field, model, device, max_len=50):
    trgs = []
    pred_trgs = []
    with torch.no_grad():
        for batch in iterator:
            src = batch.src
            trg = batch.trg
            _trgs = []
            for sentence in trg:
                tmp = []
                # Start from the first token which skips the <start> token
                for i in sentence[1:]:
                    # Targets are padded. So stop appending as soon as a padding or eos token is encountered
                    if (
                        i == trg_field.vocab.stoi[trg_field.eos_token]
                        or i == trg_field.vocab.stoi[trg_field.pad_token]
                    ):
                        break
                    tmp.append(trg_field.vocab.itos[i])
                _trgs.append([tmp])
            trgs += _trgs
            pred_trg, _ = translate_sentence_vectorized(
                src, src_field, trg_field, model, device
            )
            pred_trgs += pred_trg
    return pred_trgs, trgs, bleu_score(pred_trgs, trgs)


########################################################################
if __name__ == "__main__":
    run_test()

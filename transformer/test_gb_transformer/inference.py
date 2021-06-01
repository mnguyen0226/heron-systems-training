from main import *
from utils.preprocess import *
from utils.seq2seq import *

# This script basically for testing actual transation and analyze the attention matrix

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    """Inference script used to translate actual model

    Parameters
    ----------
    sentence:
        input tokenized German sentence
    src_field:  
        preprocessed SRC from spaCy - similar to preprocess.py 
    trg_field:
        preprocessed TRG from spaCy - similar to preprocess.py
    model:
        trained model
    device:
        cpu or gpu
    max_len:
        max length of the input sentence

    Return
    ---------
    trg_token:
        tokenized vector of the prediction
    attention:
        the attention matrix for later use
    """
    model.eval() # initialize evaluating mode 
    
    # tokenized the source sentence if it has not been tokenized (is a string)
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # append the <sos> (start of sentence) and <eos> (end of sentence) 
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    # numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    # convert it to a tensor (for PyTorch) and add a batch dim (unsqueeze)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    # create a source sentence mask
    src_mask = model.make_src_mask(src_tensor)
    
    # feed the source sentence and mask into the encoder
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # create a list to hold the output sentence, and initialized with an <sos> token
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    # while we have not hit the max length = 50
    for i in range(max_len):
        
        # convert the current output sentence predition into the tensor with a batch dim
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        # create a target sentence mask
        trg_mask = model.make_trg_mask(trg_tensor)
        
        # place the current output, encoder output and both masks into the decoder
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # get next output token prediction from decoder along with attention
        pred_token = output.argmax(2)[:,-1].item()
        
        # add prediction to current output sentence predictions
        trg_indexes.append(pred_token)

        # break if the prediction was an <eos> tokens
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    # convert the output sentence from indexes to tokens
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    # return the output sentence (with <sos> token removed) and the attention from the last layer
    return trg_tokens[1:], attention

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    """Display the attention matrix

    Parameters
    ----------
    sentence:
        input tokenized German sentence
    translation:
        translated tokenized English sentence
    attention:
        attention get from the translate_sentence()
    n_heads:
        number of heads for the attention mechanism initialized in the trained model
    n_rows:
        number of rows for 8 displays
    n_cols:
        number of columns for 8 displays
    """
    assert n_rows * n_cols == n_heads # make sure I got the correct number of heads and display matrix
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def run_inference_1():
    """Runs the inference analysis from the train set
    """
    example_idx = 8

    src = vars(train_data.examples[example_idx])['src']
    trg = vars(train_data.examples[example_idx])['trg']

    print(f'src = {src}')
    print(f'trg = {trg}')

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f'predicted trg = {translation}')

    display_attention(src, translation, attention)

def run_inference_2():
    """Runs the inference analysis from the validation set
    """
    example_idx = 6

    src = vars(valid_data.examples[example_idx])['src']
    trg = vars(valid_data.examples[example_idx])['trg']

    print(f'src = {src}')
    print(f'trg = {trg}')

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f'predicted trg = {translation}')

    display_attention(src, translation, attention)

def run_inference_3():
    """Runs the inference analysis on the test set
    """
    example_idx = 10

    src = vars(test_data.examples[example_idx])['src']
    trg = vars(test_data.examples[example_idx])['trg']

    print(f'src = {src}')
    print(f'trg = {trg}')

    translation, attention = translate_sentence(src, SRC, TRG, model, device)

    print(f'predicted trg = {translation}')

    display_attention(src, translation, attention)

if __name__ == "__main__":
    run_inference_1()
    run_inference_2()
    run_inference_3()
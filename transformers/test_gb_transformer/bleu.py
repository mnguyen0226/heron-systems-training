from torchtext.data.metrics import bleu_score
from main import *
from inference import *

"""This script calculate the BLEU values for the Transformer

About BLEU:
- BLEU is the metric that specifically designed for measuring the quality of trhe translation
- BLEU looks at the overlap in the predicted and actual target sequences in terms of their n-grams
- It will give us a number between 0 & 1 for each sequence, where 1 means there is perfect overlap - perfect translation
"""
def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    """Calculate the BLEU value

    Parameters
    ----------
    data:
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

    """    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)

def bleu():
    """Run BLEU calculation on the trained model
    """
    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

    print(f'BLEU score = {bleu_score*100:.2f}')

if __name__ == "__main__":
    bleu()
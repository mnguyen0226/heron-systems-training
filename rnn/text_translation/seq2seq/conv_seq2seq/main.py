""" 
Paper: https://arxiv.org/abs/1705.03122
About: We will not use RNN based model and implement a fully convolutional model.
- One of the downsides of RNN is that they are sequential. 
- That is, before a word is processed by the RNN, all previous words must also be processed.
- Convolutional models can be fully parallelized, which allow them to be trained much quicker.
- The model uses multiple convolutional layers in both the encoder and decoder, with attention mechanism between them

- The convolutional layer uses filter.
    These filters have a width for text and each layer 1024 filter for this project
    Each filter will slide across the sequence, from beginning to the end, looking at all 3 consecutive tokens at a time.
- The idea is that each of these 1024 filters will learn to extract different feature from the text.
    The extraction will the be used by the model, potentially as input to another convolutional layer.
    This then all be used to extract features from the srouce sentence to translate it into target language    
"""

from utils.preprocess import *
from utils.model import *

def main():
    print("Running")

if __name__ == "__main__":
    main()
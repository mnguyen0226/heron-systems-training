"""
Paper: https://arxiv.org/abs/1409.0473

Unlike the traditional statistical machine translateion, NMT aims to build a single network that can be jointly tuned to maximize the translation performace
The model proposed recently for nmt often belong to family of encoder-decoders and consist of encoder that encodes a source sentence into a fixed-length vector from which the decoder generates a translation
In this paper propose to extend the fixe-length vector by allowing the model to automatically softsearch for parts fa source sentence that are relevant to prediction a target word

The model 2 reduced some of the compression but the context vector still needs to contain all of the info about the source sentence.
The model 3 reduced some of the compression by allowing the decoder to look at enture source sentence via its hidden state at each decoding step. With Attention

How attention work?
    - First calculate an attention vectore that is the length of the src sentence
    - The attention vector has property that each element is between 0 and 1 and the entire vector sum to 1
    - We then calculate the weight sum of the source sentence hidden state to get a weighted source vectore
    - Calculate a new weighted source vectore every time-step when decoding, using it as input to our decoder RNN as well as the linear layer to make prediction
"""

def main():
    print("Running")

if __name__ == "__main__":
    main()
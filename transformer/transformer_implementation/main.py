""" 
Paper: https://arxiv.org/abs/1706.03762

About: 
- The best performing model connects with encoder and decoder through attention mechanism
- New simple neural networks architecture - Transformer - solely based on the attentio mechanism
- The Transformer generalizes well to other tasks by applying it successfully to English constituency parsing bath with large and limited traing data
- This paper intro to Multi-Head Attention
- The encoder and Decoder are made of multiple layers, with each layer consisting of Multi Head Attention and Positionwise Feedforward sublayers

Resources:
- http://nlp.seas.harvard.edu/2018/04/03/attention.html
- https://jalammar.github.io/illustrated-transformer/
- https://www.mihaileric.com/posts/transformers-attention-in-disguise/

About Transformer:
- Demonstrate that RNN and CNN are not essential for building high-performance NLP model
- Transformer achief state of the art machine translation result usig a self-attention operation
- Attention is highly-efficient opertation due to its parallelizability and runtime characteristic 
- The entire model is made up of linear layers, attention mechanishm, and normalization
"""

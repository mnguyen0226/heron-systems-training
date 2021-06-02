# Heron Systems Training

This repo contains some of elf-learning materials/projects that I did during my internship @HeronSystems - Summer 2021

## PyTorch Seq2Seq Models - NLP:

Credit: https://github.com/bentrevett/pytorch-seq2seq

- 1/ Sequence to Sequence Learning with Neural Networks
    - Paper: https://arxiv.org/abs/1409.3215
    - Architecture: LSTM
    - Code: https://github.com/mnguyen0226/heron-systems-training/tree/main/rnn/text_translation/seq2seq/seq2seq_with_nn
    ```
    python ./main.py
    python ./test_model.py
    ```
- 2/ Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
    - Paper: https://arxiv.org/abs/1406.1078
    - Architecture: RNN, GRUs
    - Code: https://github.com/mnguyen0226/heron-systems-training/tree/main/rnn/text_translation/seq2seq/statistical_machine_translation
    ```
    python ./main.py
    python ./test_model.py
    ```
- 3/ Neural Machine Translation by Jointly Learning to Align Translate
    - Paper: https://arxiv.org/abs/1409.0473
    - Architecture: Attention
    - Code: https://github.com/mnguyen0226/heron-systems-training/tree/main/rnn/text_translation/seq2seq/nmt_jointly_learning
    ```
    python ./main.py
    python ./test_model.py
    ```
- 4/ Packed Padded Sequences, Masking, Inference, and BLEU
    - Tools: Inference, BLEU
    - Code: https://github.com/mnguyen0226/heron-systems-training/tree/main/rnn/text_translation/seq2seq/packed_padded_sequences_bleu
    ```
    python ./main.py
    python ./test_model.py
    python ./inference.py
    python ./bleu.py
    ```
- 5/ Convolutional Sequence to Sequence Learning
    - Paper: https://arxiv.org/abs/1705.03122
    - Architecture: CNN
    - Code: https://github.com/mnguyen0226/heron-systems-training/tree/main/rnn/text_translation/seq2seq/conv_seq2seq
    ```
    python ./main.py
    python ./test_model.py
    python ./inference.py
    ```

## Transformers:
- 1/ Original Transfomers - Attention Is All You Need
    - Paper: https://arxiv.org/abs/1706.03762
    - Architecture: Attention, Transformers
    - Code: https://github.com/mnguyen0226/heron-systems-training/tree/main/transformer/transformer_implementation
    ```
    python ./main.py
    python ./test_model.py
    python ./inference.py
    python ./bleu.py
    ```
- 2/ Gated Transformers-XL - Stabilizing Transformers for Reinforcement Learning
    - Paper: https://arxiv.org/abs/1910.06764
    - Architecture: Attention, Transformers-XL, GRU
    - Code: https://github.com/mnguyen0226/heron-systems-training/tree/main/transformer/test_gb_transformer
    ```
    python ./main.py
    python ./test_model.py
    python ./inference.py
    python ./bleu.py
    ```
    
## Pytest API:

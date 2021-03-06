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

from utils.model import *
from utils.preprocess import *

# Define encoder and decoder
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(
    INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device
)

dec = Decoder(
    OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device
)

# Define whole Seq2Seq encapsulating model
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


def count_parameters(model):
    """Check number of training parameters

    Parameters
    ----------
    model:
        input seq2seq model

    Return
    ----------
    Total number of training parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


def initialize_weights(m):
    """Xavier uniform initialization

    Parameters
    ----------
    m:
        input model
    """
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0005

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Cross Entropy Loss Function
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    """Train by calculating losses and update parameters

    Parameters
    ----------
    model:
        input seq2seq model
    iterator:
        SRC, TRG iterator
    optimizer:
        Adam optimizer
    criterion:
        Cross Entropy Loss function
    clip:
        ?

    Return
    ----------
    epoch_loss / len(iterator)
        Loss percentage during training
    """

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    """Evaluate same as training but no gradient calculation and parameter updates

    Parameters
    ----------
    iterator:
        SRC, TRG iterator
    criterion:
        Cross Entropy Loss function

    Return
    ----------
    epoch_loss / len(iterator):
        Loss percentage during validating
    """

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    """Tells how long an epoch takes

    Parameters
    ----------
    start_time:
        start time
    end_time:
        end_time

    Return
    ----------
    Epoch run time
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1


def main():
    """Run Training and Evaluating procedure"""
    print("Running model.py")

    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "tut6-model.pt")

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
        )


if __name__ == "__main__":
    main()

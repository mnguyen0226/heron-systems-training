# Script trainining gated transformers model

import torch
import torch.nn as nn
from typing import Tuple
import math
import time


def count_parameters(model: Tuple[tuple, tuple, tuple, tuple, str]) -> int:
    """Check number of training parameters

    Parameters
    ----------
    model: [tuple, tuple, tuple, tuple, str]
        input seq2seq model

    Return
    ----------
    Total number of training parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def masked_initialize_weights(m: Tuple[tuple, tuple, tuple, tuple, str]):
    """Xavier uniform initialization

    Parameters
    ----------
    m: [tuple, tuple, tuple, tuple, str]
        input model
    """
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(
    model: Tuple[tuple, tuple, tuple, tuple, str],
    iterator: int,
    optimizer: int,
    criterion: int,
    clip: int,
) -> float:
    """Train by calculating losses and update parameters

    Parameters
    ----------
    model: [tuple, tuple, tuple, tuple, str]
        input seq2seq model
    iterator: int
        SRC, TRG iterator
    optimizer: int
        Adam optimizer
    criterion: int
        Cross Entropy Loss function
    clip: int
        Clip training process

    Return
    ----------
    epoch_loss / len(iterator): float
        Loss percentage during training
    """

    model.train()

    epoch_loss = 0

    for batch in iterator:

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


def evaluate(model, iterator: int, criterion: int) -> float:
    """Evaluate same as training but no gradient calculation and parameter updates

    Parameters
    ----------
    iterator: int
        SRC, TRG iterator
    criterion: int
        Cross Entropy Loss function

    Return
    ----------
    epoch_loss / len(iterator): float
        Loss percentage during validating
    """

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for batch in iterator:

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


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Tells how long an epoch takes

    Parameters
    ----------
    start_time:
        start time
    end_time:
        end_time

    Return
    ----------
    elapsed_mins: float
        elapse minutes
    elapsed_secs: float
        elapse seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def masked_gated_transformers_main(
    model, train_iterator, optimizer, criterion, CLIP, valid_iterator, n_epochs
) -> Tuple[float, float, float, float]:
    """Run Training and Evaluating procedure

    Return
    ----------
    train_loss: float
        training loss of the current epoch
    valid_loss: float
        validating loss of the current epoch
    math.exp(train_loss): float
        training PPL
    math.exp(valid_loss): float
        validating PPL
    """
    print(
        f"The masked gated transformers model has {count_parameters(model):,} trainable parameters"
    )

    best_valid_loss = float("inf")

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "masked-gated-tut6-model.pt")

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
        )

    return train_loss, valid_loss, math.exp(train_loss), math.exp(valid_loss)


def validate_masked_gated_transformers_model(
    model, test_iterator, criterion
) -> Tuple[float, float]:
    """Tests the trained gated transformers model with testing dataset

    Return
    ----------
    test_loss:
        Testing loss
    math.exp(test_loss):
        Testing PPL
    """
    model.load_state_dict(
        torch.load("masked-gated-tut6-model.pt", map_location=torch.device("cpu"))
    )

    test_loss = evaluate(model, test_iterator, criterion)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")

    return test_loss, math.exp(test_loss)

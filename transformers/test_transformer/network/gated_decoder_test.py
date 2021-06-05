from typing import Tuple

import numpy as np
import torch
from torch import nn

from gamebreaker.classifier.network.gated_decoder import Decoder
from gamebreaker.classifier.network.gated_encoder import Encoder


class GatedTransformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        nb_heads: int,
        max_seq_len: int,
        scale: bool,
        dropout: float,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(input_shape, nb_heads, scale, dropout)
        self.decoder = Decoder(input_shape, max_seq_len, nb_heads, scale, dropout)

    def forward(self, xs: torch.Tensor, prev_seq: torch.Tensor) -> torch.Tensor:
        encoded_xs = self.encoder(xs)
        next_probs = self.decoder(prev_seq, encoded_xs)
        return next_probs


class AutoRegTransformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        nb_heads: int,
        max_seq_len: int,
        scale: bool,
        dropout: float,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.encoder = Encoder(input_shape, nb_heads, scale, dropout)
        self.decoder = Decoder(input_shape, max_seq_len, nb_heads, scale, dropout)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        b, f, s = xs.shape
        encoded_xs = self.encoder(xs)
        outputs = torch.zeros(b, self.max_seq_len, s).to(xs.device)
        prev_seq = torch.zeros(b, f, self.max_seq_len).to(xs.device)
        for t in range(self.max_seq_len):
            new_outputs = self.decoder(prev_seq, encoded_xs)
            selected_indices = torch.argmax(new_outputs, dim=-1)
            prev_seq[list(range(b)), :, t] = xs[list(range(b)), :, selected_indices]
            outputs[:, t, :] = new_outputs

        return outputs


def binary(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Converts x to binary numbers of len(bits)

    Parameters
    ----------
    x: torch.Tensor
        Numbers to convert to binary
    bits: int
        Bit length of output

    Returns
    -------
    torch.Tensor
        Binary representations of x
    """
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def test_autoreg():
    nb_bits = 4
    sequence_len = 8
    nb_samples = 9600
    train_samples = nb_samples - 1600
    batch_size = 32
    nb_epochs = 100

    transformer = AutoRegTransformer(
        (nb_bits, sequence_len), 1, sequence_len, False, 0.0
    ).to(0)
    optim = torch.optim.Adam(transformer.parameters(), 0.001)
    loss_fcn = nn.CrossEntropyLoss()
    possible_labels = list(range(1, 2 ** nb_bits))

    permutations = torch.tensor(
        [
            np.random.choice(possible_labels, sequence_len, replace=False)
            for _ in range(nb_samples)
        ]
    )
    _, labels = torch.sort(permutations)
    inputs = binary(permutations, nb_bits).float()

    training_inputs = inputs[0:train_samples]
    training_labels = labels[0:train_samples]

    x_train = torch.reshape(
        training_inputs, (-1, batch_size, sequence_len, nb_bits)
    ).permute(0, 1, 3, 2)
    y_train = torch.reshape(training_labels, (-1, batch_size, sequence_len))

    validation_inputs = inputs[train_samples:]
    validation_labels = labels[train_samples:]

    x_val = torch.reshape(
        validation_inputs, (-1, batch_size, sequence_len, nb_bits)
    ).permute(0, 1, 3, 2)
    y_val = torch.reshape(validation_labels, (-1, batch_size, sequence_len))

    max_val_acc = 0
    for epoch in range(nb_epochs):
        for x_batch, y_batch in zip(x_train, y_train):
            optim.zero_grad()
            y_pred = transformer(x_batch.to(0))
            loss = loss_fcn(
                y_pred.view(-1, sequence_len), y_batch.view(-1).to(0).long()
            )
            loss.backward()
            optim.step()

        with torch.no_grad():
            avg_acc = torch.zeros(1).to(0)
            denom = 0
            for x_batch, y_batch in zip(x_val, y_val):
                y_pred = transformer(x_batch.to(0))
                avg_acc += torch.sum(torch.argmax(y_pred, dim=-1) == y_batch.to(0))
                denom += y_batch.shape[0] * y_batch.shape[1]
            avg_acc = avg_acc / denom
            print(f"Validation Acc: {avg_acc.item(): 0.3f}")
            if avg_acc.item() > max_val_acc:
                max_val_acc = avg_acc.item()
                if max_val_acc > 0.99:
                    break

    assert max_val_acc > 0.99


def test_gated_transformer():
    nb_bits = 4
    sequence_len = 8
    nb_samples = 1200
    train_samples = nb_samples - 200
    batch_size = 16
    nb_epochs = 100

    transformer = GatedTransformer(
        (nb_bits, sequence_len), 1, sequence_len, False, 0.0
    ).to(0)
    optim = torch.optim.Adam(transformer.parameters(), 0.01)
    loss_fcn = nn.CrossEntropyLoss()
    possible_labels = list(range(1, 2 ** nb_bits))

    permutations = torch.tensor(
        [
            np.random.choice(possible_labels, sequence_len, replace=False)
            for _ in range(nb_samples)
        ]
    )
    _, labels = torch.sort(permutations)
    inputs = binary(permutations, nb_bits).float()

    training_inputs = inputs[0:train_samples]
    training_labels = labels[0:train_samples]

    x_train = torch.zeros(
        (sequence_len * training_inputs.shape[0], sequence_len, nb_bits)
    )
    z_train = torch.zeros(
        (sequence_len * training_inputs.shape[0], sequence_len, nb_bits)
    )
    y_train = torch.zeros((sequence_len * training_labels.shape[0], 1))
    # Loop over every single sequence
    for s_idx, (x_s, y_s) in enumerate(zip(training_inputs, training_labels)):
        # Loop through the sequence
        for i_idx, (x_i, y_i) in enumerate(zip(x_s, y_s)):
            x_train[s_idx * sequence_len + i_idx, :, :] = x_s
            z_train[s_idx * sequence_len + i_idx, 0:i_idx, :] = x_s[
                torch.flip(y_s[0:i_idx], (0,)), :
            ]
            y_train[s_idx * sequence_len + i_idx, 0] = y_i

    x_train = torch.reshape(x_train, (-1, batch_size, sequence_len, nb_bits)).permute(
        0, 1, 3, 2
    )
    z_train = torch.reshape(z_train, (-1, batch_size, sequence_len, nb_bits)).permute(
        0, 1, 3, 2
    )
    y_train = torch.reshape(y_train, (-1, batch_size, 1))

    validation_inputs = inputs[train_samples:]
    validation_labels = labels[train_samples:]

    x_val = torch.zeros(
        (sequence_len * validation_inputs.shape[0], sequence_len, nb_bits)
    )
    z_val = torch.zeros(
        (sequence_len * validation_inputs.shape[0], sequence_len, nb_bits)
    )
    y_val = torch.zeros((sequence_len * validation_labels.shape[0], 1))
    # Loop over every single sequence
    for s_idx, (x_s, y_s) in enumerate(zip(validation_inputs, validation_labels)):
        # Loop through the sequence
        for i_idx, (x_i, y_i) in enumerate(zip(x_s, y_s)):
            x_val[s_idx * sequence_len + i_idx, :, :] = x_s
            z_val[s_idx * sequence_len + i_idx, 0:i_idx, :] = x_s[
                torch.flip(y_s[0:i_idx], (0,)), :
            ]
            y_val[s_idx * sequence_len + i_idx, 0] = y_i
    x_val = torch.reshape(x_val, (-1, batch_size, sequence_len, nb_bits)).permute(
        0, 1, 3, 2
    )
    z_val = torch.reshape(z_val, (-1, batch_size, sequence_len, nb_bits)).permute(
        0, 1, 3, 2
    )
    y_val = torch.reshape(y_val, (-1, batch_size, 1))

    for epoch in range(nb_epochs):
        for x_batch, z_batch, y_batch in zip(x_train, z_train, y_train):
            optim.zero_grad()
            y_pred = transformer(x_batch.to(0), z_batch.to(0))
            loss = loss_fcn(y_pred, y_batch.squeeze().to(0).long())
            loss.backward()
            optim.step()

        with torch.no_grad():
            avg_acc = torch.zeros(1).to(0)
            denom = 0
            for x_batch, z_batch, y_batch in zip(x_val, z_val, y_val):
                y_pred = transformer(x_batch.to(0), z_batch.to(0))
                avg_acc += torch.sum(
                    torch.argmax(y_pred, dim=-1) == y_batch.squeeze().to(0)
                )
                denom += y_batch.shape[0] * y_batch.shape[1]
            avg_acc = avg_acc / denom
            print(f"Validation Acc: {avg_acc.item(): 0.3f}")
            if avg_acc > 0.97:
                break

    assert avg_acc > 0.97

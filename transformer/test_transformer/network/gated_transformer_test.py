from collections import OrderedDict
from typing import Tuple

import numpy as np
import pytest
import torch
from torch import nn

from gamebreaker.classifier.network.gated_decoder import Decoder
from gamebreaker.classifier.network.gated_encoder import Encoder


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
        self.max_seq_len = max_seq_len + 1
        self.encoder = Encoder(input_shape, nb_heads, scale, dropout)
        self.decoder = Decoder(input_shape, max_seq_len + 1, nb_heads, scale, dropout)
        self.output_layers = nn.Sequential(
            OrderedDict(
                [
                    ("flatten", nn.Flatten()),
                    (
                        "linear_out",
                        nn.Linear(
                            input_shape[0] * (max_seq_len + 1), input_shape[-1] + 1, bias=False,
                        ),
                    ),
                    # ("softmax_out", nn.Softmax(dim=-1)),
                ]
            )
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        b, f, s = xs.shape
        decoder_choices = torch.zeros((b, f, s + 1)).to(xs.device)
        decoder_choices[:, :, :-1] = torch.clone(xs)
        decoder_choices[:, :, -1] = -1
        encoded_xs = self.encoder(xs)
        outputs = torch.zeros(b, self.max_seq_len, s + 1).to(xs.device)
        prev_seq = torch.zeros(b, f, self.max_seq_len).to(xs.device)
        for t in range(self.max_seq_len):
            decoder_out = self.decoder(prev_seq, encoded_xs)
            new_outputs = self.output_layers(decoder_out)
            selected_indices = torch.argmax(new_outputs, dim=-1)
            prev_seq[list(range(b)), :, t] = decoder_choices[list(range(b)), :, selected_indices]
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
    batch_size = 64
    nb_epochs = 100

    transformer = AutoRegTransformer((nb_bits, sequence_len), 1, sequence_len, False, 0.0).to(0)
    optim = torch.optim.Adam(transformer.parameters(), 0.001)
    loss_fcn = nn.CrossEntropyLoss()
    possible_labels = list(range(1, 2 ** nb_bits))

    permutations = torch.tensor(
        [np.random.choice(possible_labels, sequence_len, replace=False) for _ in range(nb_samples)]
    )
    permutations = torch.cat(
        (permutations, (2 ** nb_bits) * torch.ones(permutations.shape[0], 1)), dim=1
    ).long()
    order, labels = torch.sort(permutations)

    labels = torch.where(order <= (2 ** (nb_bits - 1)), labels, -1 * torch.ones_like(labels))
    labels = torch.where(
        order == (2 ** (nb_bits - 1)), sequence_len * torch.ones_like(labels), labels
    )

    inputs = binary(permutations[:, :-1], nb_bits).float()

    training_inputs = inputs[0:train_samples]
    training_labels = labels[0:train_samples]

    x_train = torch.reshape(training_inputs, (-1, batch_size, sequence_len, nb_bits)).permute(
        0, 1, 3, 2
    )
    y_train = torch.reshape(training_labels, (-1, batch_size, sequence_len + 1))

    validation_inputs = inputs[train_samples:]
    validation_labels = labels[train_samples:]

    x_val = torch.reshape(validation_inputs, (-1, batch_size, sequence_len, nb_bits)).permute(
        0, 1, 3, 2
    )
    y_val = torch.reshape(validation_labels, (-1, batch_size, sequence_len + 1))

    max_val_acc = 0
    for epoch in range(nb_epochs):
        train_acc = torch.zeros(1).to(0)
        train_denom = 0
        for x_batch, y_batch in zip(x_train, y_train):
            optim.zero_grad()
            y_pred = transformer(x_batch.to(0))
            batch_mask = y_batch.view(-1) > 0
            loss = loss_fcn(
                y_pred.view(-1, sequence_len + 1)[batch_mask, :],
                y_batch.view(-1)[batch_mask].to(0).long(),
            )
            loss.backward()
            optim.step()
            train_acc += torch.sum(
                torch.argmax(y_pred.view(-1, sequence_len + 1)[batch_mask, :], dim=-1)
                == y_batch.view(-1)[batch_mask].to(0)
            )
            train_denom += len(y_batch.view(-1)[batch_mask])

        with torch.no_grad():
            avg_acc = torch.zeros(1).to(0)
            denom = 0
            for x_batch, y_batch in zip(x_val, y_val):
                y_pred = transformer(x_batch.to(0))
                batch_mask = y_batch.view(-1) > 0
                avg_acc += torch.sum(
                    torch.argmax(y_pred.view(-1, sequence_len + 1)[batch_mask, :], dim=-1)
                    == y_batch.view(-1)[batch_mask].to(0)
                )
                denom += len(y_batch.view(-1)[batch_mask])
            avg_acc = avg_acc / denom
            train_acc = train_acc / train_denom
            print(
                f"[{epoch:3d}] Training Acc: {train_acc.item(): 0.3f}\tValidation Acc: {avg_acc.item(): 0.3f}"
            )
            if avg_acc.item() > max_val_acc:
                max_val_acc = avg_acc.item()
                if max_val_acc > 0.99:
                    break

    assert max_val_acc > 0.99


torch.autograd.set_detect_anomaly(True)
test_autoreg()

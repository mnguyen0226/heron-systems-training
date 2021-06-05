import os
from typing import List

import pytest
import torch

from gamebreaker.classifier.utils.training_utils import GBDataset


def test_rolling_average():
    pass


def test_model_results_logger():
    pass


class TestGBDataset:
    gpu_id = 0 if torch.cuda.is_available() else None
    batch_size = 2
    ram_size = 8
    indices = []

    cwd = os.path.realpath(__file__).split("/")[0:-3] + ["data", "Testing"]
    if os.path.realpath(__file__)[0] == "/":
        cwd[0] = f"/{cwd[0]}"

    dataset = GBDataset(
        filepath=os.path.join(*cwd),
        batch_size=batch_size,
        gpu_id=gpu_id,
        indices=indices,
        ram_size=ram_size,
    )

    def test_len(self):
        assert len(self.dataset) == self.dataset._race_data.shape[0] != 0

    def test_iter(self):
        for units, context, labels in self.dataset:
            assert units.shape[0] == self.batch_size
            assert context.shape[0] == self.batch_size
            assert labels.shape[1] == self.batch_size
            break

    def test_iter_full(self):
        counter = 0
        for ix, (_, _, _) in enumerate(self.dataset):
            counter += 1

        assert counter == len(self.dataset)

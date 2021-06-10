import numpy as np
import pytest
import torch
from adept.network import ModularNetwork
from adept.preprocess import GPUPreprocessor
from adept.utils.util import DotDict

from gamebreaker.classifier.network import CUSTOM_REGISTRY


@pytest.mark.parametrize(
    "min_acc,nb_encoders,dropout,nb_layer,linear_nb_hidden,nb_heads",
    [
        (0.95, 1, 0.0, 2, 8, 1),
        (0.95, 2, 0.0, 2, 16, 1),
        (0.95, 1, 0.5, 2, 8, 1),
        (0.95, 2, 0.5, 2, 16, 1),
        (0.95, 1, 0.0, 2, 8, 2),
        (0.95, 2, 0.0, 2, 16, 2),
        (0.95, 1, 0.5, 2, 8, 2),
        (0.95, 2, 0.5, 2, 16, 2),
    ],
)
def test_training(
    min_acc: float,
    nb_encoders: int,
    dropout: float,
    nb_layer: int,
    linear_nb_hidden: int,
    nb_heads: int,
) -> None:
    """

    Parameters
    ----------
    min_acc: float
        The desired accuracy needed to pass the test.
    nb_encoders: int
        How many encoder layers to stack together
    dropout: float
        Dropout to apply between encoders
    nb_layer: int
        Number of linear layers
    linear_nb_hidden: int
        Number of hidden units in the linear layers
    nb_heads: int
        Number of heads for multi-head attention
    Returns
    -------
    None
    """
    batch_size = 32
    feature_size = 10
    sequence_length = 100

    # The arguments needed to build Adept's ModularNetwork. The plan is to have the encoders feed
    # into a single linear output head with output size 1. This ensures that the linear layer isn't
    # adding any dependencies to the order of the sequence. However, we do have to test as a
    # regression problem.
    args = {
        "net2d": "GatedEncoder",
        "netbody": "AvgPoolingLinear",
        "head1d": "Identity1D",
        "head2d": "Identity2D",
        "nb_encoders": nb_encoders,
        "dropout": 0.0,
        "nb_layer": nb_layer,
        "linear_nb_hidden": linear_nb_hidden,
        "nb_heads": nb_heads,
    }

    # The observeration space (dimensionality of the input)
    obs_space = {"input": (feature_size, sequence_length)}

    # The dimensionality of the output space. Since we're treating this as a regression problem,
    # the output will be just a single number.
    output_space = {"out": (1,)}

    # ModularNetwork needs a GPUPreprocessor. Ignore this.
    gpu_preprocessor = GPUPreprocessor([], obs_space)

    # Create the ModularNetwork from our arguments
    network = ModularNetwork.from_args(
        DotDict(args), obs_space, output_space, gpu_preprocessor, CUSTOM_REGISTRY
    )
    network = network

    # Initialize the optimizer and loss for training
    optim = torch.optim.Adam(network.parameters(), 0.001)
    loss = torch.nn.MSELoss()

    # Track the maximum accuracy
    max_acc = 0
    for epoch_num in range(100):
        # Track the epoch's accuracy
        epoch_acc = None
        for batch_num in range(100):
            # Generate a random label between 0 and sequence length
            labels = torch.randint(sequence_length, size=(batch_size,))

            # Using those labels, generate an array of zeros and ones vectors with length
            # feature_size. The number of ones vectors in the array should equal the label.
            batch = torch.zeros(batch_size, feature_size, sequence_length)
            for ix, label in enumerate(labels):
                batch[ix, :, 0:label] += 1.0

            batch = batch.float()

            # Pass the batch through the network
            temp = network({"input": batch}, {})

            temp = temp[0]["out"].squeeze()
            temp = temp.double()
            hard_labels = torch.sigmoid(temp)

            labels = torch.div(labels, sequence_length)

            # Accuracy is treated as a classification problem
            accuracy = (1 - torch.abs(hard_labels - labels)).float().detach().cpu()
            # Calculate loss and backprop
            batch_loss = loss(hard_labels, labels.double())

            optim.zero_grad()
            batch_loss.backward()
            optim.step()

            if epoch_acc is None:
                epoch_acc = accuracy
            else:
                epoch_acc = torch.cat((epoch_acc, accuracy))
        if torch.mean(epoch_acc) > max_acc:
            max_acc = torch.mean(epoch_acc)

        if max_acc >= min_acc:
            break

    # Assert that we've met the minimum accuracy requirement
    assert max_acc >= min_acc, f"Expected acc: {min_acc}, Actual: {max_acc}"


@pytest.mark.parametrize(
    "nb_encoders,max_diff,nb_layer,linear_nb_hidden,nb_heads",
    [
        (1, 1e-5, 2, 8, 1),
        (2, 1e-5, 2, 16, 1),
        (1, 1e-5, 2, 8, 1),
        (2, 1e-5, 2, 16, 1),
        (1, 1e-5, 2, 8, 2),
        (2, 1e-5, 2, 16, 2),
        (1, 1e-5, 2, 8, 2),
        (2, 1e-5, 2, 16, 2),
    ],
)
def test_order_invariance(
    nb_encoders: int,
    max_diff: float,
    nb_layer: int,
    linear_nb_hidden: int,
    nb_heads: int,
) -> None:
    """

    Parameters
    ----------
    nb_encoders: int
        How many encoder layers to stack together
    max_diff: float
        How different the output of the attention layers can be from each other
    nb_layer: int
        Number of linear layers to add to the model
    linear_nb_hidden: int
        Number of hidden units for each of the linear layers
    nb_heads: int
        Number of heads for multihead attention

    Returns
    -------
    None
    """
    batch_size = 32
    feature_size = 10
    sequence_length = 100

    # The arguments needed to build Adept's ModularNetwork. The plan is to have the encoders feed
    # into a single linear output head with output size 1. This ensures that the linear layer isn't
    # adding any dependencies to the order of the sequence. However, we do have to test as a
    # regression problem.
    args = {
        "net2d": "GatedEncoder",
        "netbody": "AvgPoolingLinear",
        "head1d": "Identity1D",
        "head2d": "Identity2D",
        "nb_encoders": nb_encoders,
        "dropout": 0.0,
        "nb_layer": nb_layer,
        "linear_nb_hidden": linear_nb_hidden,
        "nb_heads": nb_heads,
    }

    # The observeration space (dimensionality of the input)
    obs_space = {"input": (feature_size, sequence_length)}

    # The dimensionality of the output space. Since we're treating this as a regression problem,
    # the output will be just a single number.
    output_space = {"out": (1,)}

    # ModularNetwork needs a GPUPreprocessor. Ignore this.
    gpu_preprocessor = GPUPreprocessor([], obs_space)

    # Create the ModularNetwork from our arguments
    network = ModularNetwork.from_args(
        DotDict(args), obs_space, output_space, gpu_preprocessor, CUSTOM_REGISTRY
    )
    network = network
    # Test that the model meets the sequence order invariance requirement
    with torch.no_grad():
        network.eval()

        # Generate a test batch
        labels = torch.randint(sequence_length, size=(batch_size,))
        batch = torch.zeros(batch_size, feature_size, sequence_length)
        for ix, label in enumerate(labels):
            batch[ix, :, 0:label] += 1.0

        batch = batch.float()

        # Using the test batch, shuffle the sequence order s.t. the features still correspond to the
        # proper label (all the vectors will still be exclusively 0's or exclusively 1's)
        rand_indices = torch.randperm(sequence_length)
        shuffled_batch = torch.clone(batch[:, :, rand_indices])

        # Pass both batches through the network
        pred = torch.flatten(network({"input": batch}, {})[0]["out"].squeeze(-1))
        s_pred = torch.flatten(
            network({"input": shuffled_batch}, {})[0]["out"].squeeze(-1)
        )
        assert all(
            np.isclose(pred, s_pred)
        ), f"Order Invariance Failed! Model differs by {torch.max(torch.abs(pred - s_pred))}"

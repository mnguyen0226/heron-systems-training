# Copyright (C) 2021 Heron Systems, Inc.
import os
import pprint
import time

import numpy as np
import torch
from adept.utils.util import DotDict
from pysc2.lib import units

from gamebreaker import unit_data
from gamebreaker.classifier.utils.common_utils import get_network
from gamebreaker.classifier.utils.common_utils import get_old_args
from gamebreaker.classifier.utils.common_utils import get_path_from_args
from gamebreaker.classifier.utils.dataset_utils import read_episode
from gamebreaker.env.base.obs_utils import cat_to_bin
from gamebreaker.env.base.obs_utils import get_proc_unit
from gamebreaker.env.base.obs_utils import get_raw_unit
from gamebreaker.search.neural_input_optimizer import GreedySearchSynthesizer
from gamebreaker.search.neural_input_optimizer import RandomSearchSynthesizer


if __name__ == "__main__":
    args = {
        "logdir": "/media/glacial/data/hs/gb_winprob/refactored_models/",
        "tag": "agents",
        "gpu_id": 0,
    }

    args = DotDict(args)

    # Get the path to the save directory
    model_path = get_path_from_args(args)

    # Retrieve the previous model's arguments
    args = get_old_args(args, model_path)

    # Build the network
    net = get_network(args, model_path)
    net.eval()
    path = "/media/glacial/data/hs/gb_winprob/Data/proc_dataset_v3/Testing/0"

    with torch.no_grad():
        test_games = sorted(os.listdir(path))
        filename = os.path.join(path, test_games[0])

        all_units, upgrades, labels = read_episode(filename)

        all_units = all_units.float().to(args.gpu_id)
        upgrades = upgrades.float().to(args.gpu_id)

    # Check that unit conversion works properly
    proc_units = all_units.cpu().numpy()
    raw_units = get_raw_unit(proc_units, 64, 64)
    proc_units_rec = get_proc_unit(raw_units, 64, 64)

    if np.max(np.abs(proc_units - proc_units_rec)) > 1e-4:
        raise RuntimeError("Error!")

    # Build the synthesizer
    synthesizer_type = "random"

    tic = time.time()

    if synthesizer_type == "random":
        army_synthesizer = RandomSearchSynthesizer(
            net,
            "cuda:0",
            unit_data.available_units(units.Terran),
            minerals=1000,
            gas=1000,
            units=20,
            map_size=(64, 64),
            max_evals=20,
        )
    elif synthesizer_type == "greedy":
        army_synthesizer = GreedySearchSynthesizer(
            net,
            "cuda:0",
            unit_data.available_units(units.Terran),
            minerals=1000,
            gas=1000,
            units=20,
            map_size=(64, 64),
            beam_size=2,
            partial_army=[{"pos": (None, None), "unit_type": units.Terran.Marine}],
        )
    else:
        raise ValueError("Invalid synthesizer type.")

    # TODO: make this interface more intuitive, so that we can separate evaluation logic
    # from army search logic.
    # By default, this will run random search, evaluating 1000 candidates.
    # Selecting [0] in all_units and upgrades below picks the first timestep. In particular:
    # all_units[0].shape == [n_units, 512]
    # upgrades[0].shape == [192]
    results = army_synthesizer.synthesize(all_units[0].cpu().numpy(), upgrades[0].cpu().numpy())

    toc = time.time()

    pprint.pprint(results)

    print(f"Time taken: {(toc - tic):.6f} s.")

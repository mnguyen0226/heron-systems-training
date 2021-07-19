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

# from gamebreaker.search.neural_input_optimizer_minh import GreedySearchSynthesizer
from gamebreaker.search.neural_input_optimizer_minh import RandomSearchSynthesizer


if __name__ == "__main__":
    # log parameters
    args = {
        # "logdir": "/media/glacial/data/hs/gb_winprob/refactored_models/",
        "logdir": "/media/banshee/gb_winprob/autobots/",
        "tag": "DASH150", # DASH150
        "gpu_id": 0,
    }

    # Run argument
    args = DotDict(args)

    # Get the path to the save directory - grab the model's path
    model_path = get_path_from_args(args)

    # Retrieve the previous model's arguments - grab the model from path
    args = get_old_args(args, model_path)

    # Build the network
    net = get_network(args, model_path)
    net.eval()
    # path = "/media/glacial/data/hs/gb_winprob/Data/proc_dataset_v3/Testing/0" # Karthik
    # path = "/media/banshee/gb_winprob/Data/shuffled_dataset_v3/Testing/agents/0/000"
    # path = "/media/banshee/gb_winprob/Data/raw_dataset_v3/Testing/0"
    path = "/media/banshee/gb_winprob/Data/proc_dataset_v3/Training/1000"


    # this is not a training phase so running search on the train agent
    with torch.no_grad():
        test_games = sorted(os.listdir(path))  # get the set of test game
        filename = os.path.join(path, test_games[0])

        # all_units, upgrades, labels = read_episode(
        #     filename
        # )  # Q: from pre-run game, get all unit, upgrades, and labels

        # Minh's Modification
        all_units, upgrades, _, labels, _, _, _, _, _, _ = read_episode(
            filename
        ).values()

        # all_units = all_units.float().to(args.gpu_id)  # Q: What is this suppose to be?
        # upgrades = upgrades.float().to(args.gpu_id) # Q: What is this suppose to be?

    # Check that unit conversion works properly
    # proc_units = all_units.cpu().numpy()
    # import numpy as np
    # proc_units = np.array(all_units)
    # raw_units = get_raw_unit(proc_units, 64, 64)
    # print(f"TESTING type all_units {type(raw_units)}")
    # proc_units_rec = get_proc_unit(raw_units, 64, 64)

    # if (
    #     np.max(np.abs(proc_units - proc_units_rec)) > 1e-4
    # ):  # just check if the conversion is close enough
    #     raise RuntimeError("Error!")

    # Build the synthesizer - This can be change
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

    # TODO: make this interface more intuitive, so that we can separate evaluation logic
    # from army search logic.
    # By default, this will run random search, evaluating 1000 candidates.
    # Selecting [0] - the top in all_units and upgrades below picks the first timestep. In particular:
    # all_units[0].shape == [n_units, 512]
    # upgrades[0].shape == [192]
    # results = army_synthesizer.synthesize(all_units[0].cpu().numpy(), upgrades[0].cpu().numpy())

    results = army_synthesizer.synthesize(np.array(all_units)[0], np.array(upgrades)[0])

    toc = time.time()

    pprint.pprint(results)

    print(f"Time taken: {(toc - tic):.6f} s.")
    print("FINISH RUNNING")
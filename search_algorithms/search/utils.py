import random

import numpy as np
import torch

from gamebreaker.env.base.obs_idx import ObsIdx


class Edge:
    def __init__(self, net, state):
        """
        Parameters
        ----------
        net: Class
            Class with a forward() method that returns win probability
        state: torch.Tensor
            State space at the search timestep
        """
        self.network = net
        indices = [
            ObsIdx.unit_type_bit0,
            ObsIdx.unit_type_bit1,
            ObsIdx.unit_type_bit2,
            ObsIdx.unit_type_bit3,
            ObsIdx.unit_type_bit4,
            ObsIdx.unit_type_bit5,
            ObsIdx.unit_type_bit6,
            ObsIdx.unit_type_bit7,
            ObsIdx.unit_type_bit8,
            ObsIdx.alliance_bit0,
            ObsIdx.alliance_bit1,
            ObsIdx.alliance_bit2,
            ObsIdx.x,
            ObsIdx.y,
        ]
        state = torch.index_select(
            torch.tensor(state), 1, torch.tensor([int(i) for i in indices]).long()
        )
        # create a tensor of zeros and assign enemy state to the first few
        enemy = torch.split(state, 2)[1]
        self.enemy_state = torch.zeros((512, len(state[0])))
        self.enemy_state[: len(enemy), :] = enemy

    def eval(self, units):
        """
        Returns win prob given army compositions
        """
        new_state = self.process(units)
        w = self.network.forward(new_state)
        return (w,)

    def process(self, units):
        """
        applies unit results to state space format
        unit: {"unit_type": unit, "pos": self.position(), "quantity": 1}
        """

        def get_bin(x):
            return format(x, "b").zfill(9)

        out = []
        for unit in units:
            inner = []
            # unit id
            for i in get_bin(unit["unit_type"]):
                inner.append(int(i))
            # alliance - friendly
            inner.append(1)
            inner.append(0)
            inner.append(0)
            # x pos
            inner.append(unit["pos"][0])
            # y pos
            inner.append(unit["pos"][1])
            out.append(np.array(inner))

        ally = torch.zeros((512, len(out[0])))
        ally[: len(out), :] = torch.tensor(out)

        return torch.cat((ally, self.enemy_state), 0)


def iter_bounds(selector):
    """
    Individual creation function that takes each state variable's bounds
    """
    units = selector.select()
    ind = []
    for unit in units:
        ind.append(unit["unit_type"])
        ind.append(unit["pos"][0])
        ind.append(unit["pos"][1])
    return ind


def mutate(ind, selector):
    unit_idx = random.choice(range(len(ind)))
    if unit_idx % 3 == 0 or unit_idx == 0:
        available_units = selector.available_units
        ind[unit_idx] = random.choice(available_units)
    elif unit_idx % 3 == 1:
        x_pos = selector.x_area
        ind[unit_idx] = int((x_pos[1] - x_pos[0] + 1) * random.random() + x_pos[0])
    elif unit_idx % 3 == 2:
        y_pos = selector.x_area
        ind[unit_idx] = int((y_pos[1] - y_pos[0] + 1) * random.random() + y_pos[0])
    return ind


def format_ind(state):
    out = []
    for idx in range(int(len(state) / 3)):
        unit_dict = {
            "unit_type": state[idx * 3],
            "pos": (state[idx * 3 + 1], state[idx * 3 + 2]),
            "quantity": 1,
        }
        out.append(unit_dict)
    return out

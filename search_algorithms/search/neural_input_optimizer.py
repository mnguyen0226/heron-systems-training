# Copyright (C) 2021 Heron Systems, Inc.
import copy
from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from pysc2.lib.features import PlayerRelative
from pysc2.lib.named_array import NamedNumpyArray

from gamebreaker import unit_data
from gamebreaker.env.base.obs_idx import ObsIdx
from gamebreaker.env.base.obs_utils import FIXED_UNIT_COUNT
from gamebreaker.env.base.obs_utils import get_proc_unit
from gamebreaker.env.base.obs_utils import get_raw_unit
from gamebreaker.env.base.obs_utils import raw_empty_units
from gamebreaker.env.base.obs_utils import raw_is_empty
from gamebreaker.selector.army_selector import build_constraints
from gamebreaker.selector.army_selector import GeneralUniformArmySelector
from gamebreaker.selector.army_selector import to_raw


class _NeuralFloatInputOptimizer(object):
    def __init__(self, net):
        self.net = net

    def _initialize_floats(self, population, n_self_units, safe_spawn):
        # Initialize starting locations for all "self" units
        for member, cur_n_self_units in zip(population, n_self_units):
            member[:cur_n_self_units, "x"] = np.random.randint(
                low=safe_spawn[0][0], high=safe_spawn[1][0], size=(cur_n_self_units,)
            )
            member[:cur_n_self_units, "y"] = np.random.randint(
                low=safe_spawn[0][1], high=safe_spawn[1][1], size=(cur_n_self_units,)
            )

        return population

    def _gradient_mask(self, x, n_self_units):
        ret = torch.zeros_like(x)
        for ix, (cur_x, cur_n_self_units) in enumerate(zip(x, n_self_units)):
            ret[ix, ObsIdx.x, :cur_n_self_units] = 1.0
            ret[ix, ObsIdx.y, :cur_n_self_units] = 1.0
        return ret

    def _sample_projector(self, data, n_self_units, safe_spawn_preproc):
        for member, cur_n_self_units in zip(data, n_self_units):
            member[ObsIdx.x, :cur_n_self_units] = torch.clamp(
                member[ObsIdx.x, :cur_n_self_units],
                safe_spawn_preproc[0][0],
                safe_spawn_preproc[1][0],
            )
            member[ObsIdx.y, :cur_n_self_units] = torch.clamp(
                member[ObsIdx.y, :cur_n_self_units],
                safe_spawn_preproc[0][1],
                safe_spawn_preproc[1][1],
            )
        return data

    def batch_evaluate(
        self,
        population,
        n_self_units,
        n_non_self_units,
        safe_spawn,
        upgrades,
        n_inits=1,
        device="cpu",
        n_iters=0,
        map_size=(64, 64),
        shared_environment=False,
    ):
        # population: np x 512 x 130 (ish)
        # We assume that in each population member, units are arranged as self_units,
        # followed by non_self_units.

        # We're not learning network parameters; rather, we're optimizing over the inputs
        # to the network.
        for p in self.net.parameters():
            p.requires_grad = False

        # Put together the initial list of inputs to the network, that we'll iterate on
        if shared_environment:
            x = []
            for ix in range(n_inits):
                raw_cur_x = self._initialize_floats(population, n_self_units, safe_spawn)

                # Batch together all self units, and process these
                self_units = get_proc_unit(
                    np.vstack([rx[:ns] for rx, ns in zip(raw_cur_x, n_self_units)]), *map_size
                ).astype(np.float32)

                # Take a look at the shared environment; in particular, take the shared
                # environment associated with the least number of self units created.
                shared_env_ix = np.argmin(n_self_units)
                shared_env = get_proc_unit(
                    raw_cur_x[shared_env_ix][n_self_units[shared_env_ix] :], *map_size
                )

                # Ensure that too many units weren't created.
                max_self_units = np.max(n_self_units)
                max_trailing_empty_units = max_self_units + shared_env.shape[1] - FIXED_UNIT_COUNT
                trailing_proc_units = shared_env[:, -max_trailing_empty_units:]
                if np.count_nonzero(trailing_proc_units) != 0:
                    raise ValueError(
                        "Couldn't find enough empty units in the shared environment; "
                        "are you sure that shared_environment=True is the case?"
                    )

                # Combine
                expanded = []
                offset = 0
                for pix in range(len(n_self_units)):
                    cur_self_units = self_units[:, offset : offset + n_self_units[pix]]
                    cur_env = np.hstack([cur_self_units, shared_env])[:, :FIXED_UNIT_COUNT]
                    offset += n_self_units[pix]
                    expanded.append(cur_env[None])

                x.append(np.concatenate(expanded))
        else:
            x = []
            for ix in range(n_inits):
                raw_cur_x = self._initialize_floats(population, n_self_units, safe_spawn)
                cur_x = get_proc_unit(raw_cur_x, *map_size).astype(np.float32)
                # cur_x = torch.from_numpy(cur_x).to(device)

                x.append(cur_x)

        x = torch.from_numpy(np.concatenate(x, axis=0)).to(device)
        x = torch.nn.Parameter(x, requires_grad=True)

        # Create separate optimizers per population member
        n_population = x.shape[0]
        dummy = torch.nn.Parameter(torch.zeros(x.shape[1:]).to(device), requires_grad=True)
        optimizers = [torch.optim.Adam([dummy], lr=1e-2) for ix in range(n_population)]

        upgrades_exp = torch.tensor([upgrades for _ in range(x.shape[0])])

        # Mask out parts of the gradient that we shouldn't be updating; i.e., these are
        # integer features and features of non-self units.
        grad_mask = self._gradient_mask(x, n_self_units).to(device)

        # TODO: this is hacky
        # Safe spawn, in preprocessed units
        safe_spawn_preproc = [safe_spawn[0] / map_size[0], safe_spawn[1] / map_size[1]]

        # Maximize the win-head of the network
        loss = -self.net.forward({"units": x.to(device), "upgrades": upgrades_exp.to(device)}, {})[
            0
        ]["win"]

        for iteration in range(n_iters):
            for ix in range(n_population):
                # Compute gradients
                loss[ix].backward(retain_graph=True)

                # Copy the relevant data over to dummy so that the optimizer can provide
                # the correct update.
                dummy.data = x[ix]
                dummy.grad = x.grad[ix]

                # Apply the gradient mask.
                dummy.grad *= grad_mask[ix]

                # Take an optimization step
                optimizers[ix].step()
                x.data[ix] = dummy.data

            # Ensure that units don't veer outside of where they're supposed to be
            x.data = self._sample_projector(x.data, n_self_units, safe_spawn_preproc).to(device)

            # We compute the loss at the end, so that setting n_iters = 0 still results
            # in the loss being computed.
            loss = -self.net.forward(
                {"units": x.to(device), "upgrades": upgrades_exp.to(device)}, {}
            )[0]["win"]

        return {
            "optimized_samples": x.detach().cpu().numpy(),
            "evaluations": -np.squeeze(loss.detach().cpu().numpy()),
        }


class ArmySynthesizer(ABC):
    OPTIMIZE_FLOAT_IXS = [ObsIdx.x, ObsIdx.y]

    def __init__(
        self,
        net,
        device,
        available_units: List[
            Union[unit_data.units.Terran, unit_data.units.Protoss, unit_data.units.Zerg],
        ],
        x_area: Tuple[int, int] = (0, 10),
        y_area: Tuple[int, int] = (0, 10),
        minerals: int = np.Infinity,
        gas: int = np.Infinity,
        supply: int = np.Infinity,
        units: int = np.Infinity,
        map_size: tuple = (64, 64),
        seed: int = None,
    ):
        self.net = net.to(device)

        self.device = device
        self.available_units = available_units
        self.x_area = x_area
        self.y_area = y_area
        self.minerals = minerals
        self.gas = gas
        self.supply = supply
        self.units = units
        self.map_size = map_size
        self.seed = seed

        # Army evaluator
        self._evaluator = _NeuralFloatInputOptimizer(
            self.net,
        )

        # We don't care about x_max, y_max here
        self._index_names = None

    def _create_scenario(self, raw_self_units, raw_non_self_units):
        raw_scenario = np.concatenate([raw_self_units, raw_non_self_units])

        # NOTE: raw_scenario is now a numpy array rather than a NamedNumpyArray
        # Ensure that raw_scenario has the right number of units. We assume that
        # the tail units are blank.
        too_many_units_created = False
        if raw_scenario.shape[0] < FIXED_UNIT_COUNT:
            cur_empty_units = raw_empty_units(
                FIXED_UNIT_COUNT - raw_scenario.shape[0],
                index_names=self._index_names,
            )
            raw_scenario = np.concatenate([raw_scenario, cur_empty_units])
        elif raw_scenario.shape[0] > FIXED_UNIT_COUNT:
            # Look at the tail values of the current raw scenario
            tail = raw_scenario[-(raw_scenario.shape[0] - FIXED_UNIT_COUNT) :]

            # If the tail purely consists of empty units, then we're free to just take
            # the leading portion of the army.
            if raw_is_empty(tail, self._index_names):
                raw_scenario = raw_scenario[:FIXED_UNIT_COUNT]
            # Otherwise, we know that too many units exist, which isn't good. So,
            # mark this case and proceed.
            else:
                too_many_units_created = True

        # If the scenario doesn't have too many units, then we have a valid scenario.
        if not too_many_units_created:
            return raw_scenario

    def evaluate_armies(self, samples, raw_non_self_units, safe_spawn, upgrades):
        # Use the samples to create scenarios that we're interested in evaluating
        raw_scenarios_to_eval = []
        n_self_units = []
        for sample in samples:
            cur_raw_self_units = to_raw(sample, self._index_names)
            raw_scenarios_to_eval.append(
                self._create_scenario(cur_raw_self_units, raw_non_self_units)
            )
            n_self_units.append(cur_raw_self_units.shape[0])

        # Check that all scenarios have the same shape
        assert len(set([s.shape for s in raw_scenarios_to_eval])) == 1

        # Batch the scenarios and evaluate away
        return list(
            self._evaluator.batch_evaluate(
                [NamedNumpyArray(s, [None, self._index_names]) for s in raw_scenarios_to_eval],
                n_self_units,
                [raw_non_self_units.shape[0] for _ in range(len(n_self_units))],
                safe_spawn,
                upgrades,
                device=self.device,
            )["evaluations"]
        )

    def synthesize(self, initial_state, upgrades):
        # Re-arrange scenario and compute number of not-self and self units.
        raw_initial_state = get_raw_unit(initial_state, *self.map_size)
        raw_non_self_units = raw_initial_state[
            [
                ix
                for ix in range(raw_initial_state.shape[0])
                if raw_initial_state[ix].alliance != PlayerRelative.SELF
            ]
        ]

        # TODO: right now, we're inferring safe spawn regions based on the scenario. The
        # way we do this is by computing the minimal bounding box that covers the self
        # units. We instead should have a method that computes safe spawn regions.
        raw_self_units = raw_initial_state[
            [
                ix
                for ix in range(raw_initial_state.shape[0])
                if raw_initial_state[ix].alliance == PlayerRelative.SELF
            ]
        ]
        safe_spawn = np.array([(unit.x, unit.y) for unit in raw_self_units])
        safe_spawn = (np.min(safe_spawn, axis=0), np.max(safe_spawn, axis=0) + 1)

        if not self._index_names:
            self._index_names = list(
                [d for d in raw_non_self_units._index_names if type(d) == dict][0]
            )

        # Run optimization
        return self.optimize(raw_non_self_units, safe_spawn, upgrades)

    @abstractmethod
    def optimize(self, raw_non_self_units, safe_spawn, upgrades):
        ...


class RandomSearchSynthesizer(ArmySynthesizer):
    def __init__(
        self,
        net,
        device,
        available_units: List[
            Union[unit_data.units.Terran, unit_data.units.Protoss, unit_data.units.Zerg],
        ],
        x_area: Tuple[int, int] = (0, 10),
        y_area: Tuple[int, int] = (0, 10),
        minerals: int = np.Infinity,
        gas: int = np.Infinity,
        supply: int = np.Infinity,
        units: int = np.Infinity,
        map_size: tuple = (64, 64),
        max_evals: int = 1000,
        seed: int = None,
    ):
        super().__init__(
            net,
            device,
            available_units,
            x_area=x_area,
            y_area=y_area,
            minerals=minerals,
            gas=gas,
            supply=supply,
            units=units,
            map_size=map_size,
            seed=seed,
        )

        self._gus = GeneralUniformArmySelector(
            self.available_units,
            x_area=x_area,
            y_area=y_area,
            minerals=minerals,
            gas=gas,
            supply=supply,
            units=units,
            seed=seed,
        )

        self.max_evals = max_evals

    def optimize(self, raw_non_self_units, safe_spawn, upgrades):
        # Generate the desired number of random samples
        samples = [self._gus.select() for _ in range(self.max_evals)]

        # Evaluate the samples
        evals = self.evaluate_armies(samples, raw_non_self_units, safe_spawn, upgrades)

        # Sort the samples by performance
        return sorted([(sample, ev) for sample, ev in zip(samples, evals)], key=lambda x: -x[1])


class GreedySearchSynthesizer(ArmySynthesizer):
    def __init__(
        self,
        net,
        device,
        available_units: List[
            Union[unit_data.units.Terran, unit_data.units.Protoss, unit_data.units.Zerg],
        ],
        partial_army=None,
        beam_size=1,
        x_area: Tuple[int, int] = (0, 10),
        y_area: Tuple[int, int] = (0, 10),
        minerals: int = np.Infinity,
        gas: int = np.Infinity,
        supply: int = np.Infinity,
        units: int = np.Infinity,
        map_size: tuple = (64, 64),
        seed: int = None,
    ):
        super().__init__(
            net,
            device,
            available_units,
            x_area=x_area,
            y_area=y_area,
            minerals=minerals,
            gas=gas,
            supply=supply,
            units=units,
            map_size=map_size,
            seed=seed,
        )

        constraints = build_constraints(
            available_units, minerals=minerals, gas=gas, supply=supply, units=units
        )
        self._A = np.array([x for x, y in constraints], dtype=np.float32)
        self._b = np.array([y for x, y in constraints], dtype=np.float32)

        self.partial_army = partial_army
        self.beam_size = beam_size

    def _is_army_within_budget(self, army):
        if army is None:
            return True

        vec = np.array(army, dtype=np.float32)

        # TODO: this may potentially be numerically unstable; we should really be multiplying _A and
        # _b by a constant to make all entries integers the same way we do this in the
        # GeneralUniformSelector codebase.
        satisfied = self._A.dot(vec) <= self._b

        return np.all(satisfied)

    def _fringe(self, army):
        if not self._is_army_within_budget(army):
            return

        ret = []

        for ix, unit in enumerate(self.available_units):
            # Note that candidate_army is a tuple
            candidate_army = list(army)
            candidate_army[ix] += 1
            candidate_army = tuple(candidate_army)

            if self._is_army_within_budget(candidate_army):
                ret.append(candidate_army)

        return ret

    def _army_to_tuple(self, army):
        ret = {unit: 0 for unit in self.available_units}

        if army:
            for unit in army:
                ret[unit["unit_type"]] += 1

        return tuple(ret.values())

    def _tuple_to_army(self, army):
        ret = []
        for ix, unit in enumerate(self.available_units):
            if army[ix] > 0:
                ret.extend([{"pos": (None, None), "unit_type": unit} for _ in range(army[ix])])
        return ret

    def optimize(self, raw_non_self_units, safe_spawn, upgrades):
        fringe = self._fringe(self._army_to_tuple(self.partial_army))

        beam = []
        solutions = []

        while len(fringe) > 0:
            # Evaluate each element of the fringe
            armies = [self._tuple_to_army(t) for t in fringe]
            evals = self.evaluate_armies(armies, raw_non_self_units, safe_spawn, upgrades)

            # Pick the best armies in the fringe and set this to be the beam
            beam = sorted([(sample, ev) for sample, ev in zip(fringe, evals)], key=lambda x: -x[1])[
                : self.beam_size
            ]

            # Determine the new fringe
            fringe = []
            for beam_element, ev in beam:
                cur_fringe = self._fringe(beam_element)

                # If the current beam element (i.e., army) has a fringe, then append it to the
                # fringe list.
                if cur_fringe:
                    fringe.extend(cur_fringe)

                # Otherwise, the current army is pareto optimal; so, add it to the solution set.
                else:
                    solutions.append((self._tuple_to_army(beam_element), ev))

            # Remove duplicates across the fringe
            fringe = list(set(fringe))

        return sorted(solutions, key=lambda x: -x[1])

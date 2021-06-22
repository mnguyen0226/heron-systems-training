# Copyright (C) 2021 Heron Systems, Inc.
from collections import namedtuple
from enum import Enum
from typing import List, Union

import numpy as np

from gus.selector import GeneralUniformSelector, _get_rng


class TerranUnits(Enum):
    Marine = 0
    Marauder = 1
    Reaper = 2
    SiegeTank = 3
    Hellion = 4
    Cyclone = 5
    Thor = 6


class ProtossUnits(Enum):
    Zealot = 0
    Stalker = 1
    Sentry = 2
    Adept = 3
    Immortal = 4
    Colossus = 5
    HighTemplar = 6


class ZergUnits(Enum):
    Zergling = 0
    Baneling = 1
    Roach = 2
    Hydralisk = 3
    Ultralisk = 4
    Infestor = 5


_ALL_UNITS = [u for u in TerranUnits] + [u for u in ProtossUnits] + [u for u in ZergUnits]


class Units(object):
    def __init__(self) -> None:
        self.Terran = TerranUnits
        self.Protoss = ProtossUnits
        self.Zerg = ZergUnits


units = Units()
GameUnit = Union[TerranUnits, ProtossUnits, ZergUnits]


def all_units() -> List[GameUnit]:
    """Returns a set of all the units

    Returns
    -------
    The set of Terran, Protoss, and Zerg units
    """
    return set([u for u in units.Terran] + [u for u in units.Protoss] + [u for u in units.Zerg])


def available_units(races: Union[units.Terran, units.Protoss, units.Zerg]) -> List[GameUnit]:
    """Returns a list of the available units

    Parameters
    ----------
    races : The races to check available units for

    Returns
    -------
    List of all the available units for the given races
    """
    if type(races) not in [list, tuple]:
        races = [races]
    return [u for u in _ALL_UNITS if type(u) in races]


# TODO: fields to incorporate: game_speed, armor, {ground,air}_attack, {ground,air}_dps,
# bonus_dps, attack_mod, speed, range, sight
UnitMetadata = namedtuple(
    "UnitMetadata", ["supply", "minerals", "gas", "health", "shield", "range"],
)

_UNIT_DATA = {
    # Protoss
    units.Protoss.Zealot: UnitMetadata(
        supply=2, minerals=100, gas=0, health=100, shield=50, range=0.1
    ),
    units.Protoss.Sentry: UnitMetadata(
        supply=2, minerals=50, gas=100, health=40, shield=40, range=5
    ),
    units.Protoss.Stalker: UnitMetadata(
        supply=2, minerals=125, gas=50, health=80, shield=80, range=6
    ),
    units.Protoss.Adept: UnitMetadata(
        supply=2, minerals=100, gas=25, health=70, shield=70, range=4
    ),
    units.Protoss.Immortal: UnitMetadata(
        supply=4, minerals=275, gas=100, health=200, shield=100, range=6
    ),
    units.Protoss.Colossus: UnitMetadata(
        supply=6, minerals=300, gas=200, health=200, shield=150, range=7
    ),
    units.Protoss.HighTemplar: UnitMetadata(
        # NOTE: attack range is 6, but psionic storm range is 9
        supply=2,
        minerals=50,
        gas=150,
        health=40,
        shield=40,
        range=6,
    ),
    # Terran
    units.Terran.Marine: UnitMetadata(supply=1, minerals=50, gas=0, health=45, shield=0, range=5),
    units.Terran.Marauder: UnitMetadata(
        supply=2, minerals=100, gas=25, health=125, shield=0, range=6
    ),
    units.Terran.Reaper: UnitMetadata(supply=1, minerals=50, gas=50, health=60, shield=0, range=5),
    # NOTE: attack range for siegetank is 13 in siege mode
    units.Terran.SiegeTank: UnitMetadata(
        supply=3, minerals=150, gas=125, health=175, shield=0, range=7
    ),
    units.Terran.Hellion: UnitMetadata(supply=2, minerals=100, gas=0, health=90, shield=0, range=5),
    units.Terran.Cyclone: UnitMetadata(
        supply=3, minerals=150, gas=100, health=120, shield=0, range=5
    ),
    units.Terran.Thor: UnitMetadata(
        supply=6, minerals=300, gas=200, health=400, shield=0, range=10
    ),
    # Zerg
    units.Zerg.Zergling: UnitMetadata(
        supply=0.5, minerals=25, gas=0, health=35, shield=0, range=0.1
    ),
    units.Zerg.Baneling: UnitMetadata(
        supply=0.5, minerals=25, gas=25, health=30, shield=0, range=2.2
    ),
    units.Zerg.Roach: UnitMetadata(supply=2, minerals=75, gas=25, health=145, shield=0, range=4),
    units.Zerg.Hydralisk: UnitMetadata(
        supply=2, minerals=100, gas=50, health=90, shield=0, range=5
    ),
    units.Zerg.Ultralisk: UnitMetadata(
        supply=6, minerals=300, gas=200, health=500, shield=0, range=1
    ),
    units.Zerg.Infestor: UnitMetadata(
        supply=2, minerals=100, gas=150, health=90, shield=0, range=0
    ),
}


def unit_data(unit: GameUnit) -> UnitMetadata:
    """Returns the unit data for the input unit

    Parameters
    ----------
    unit : the unit to get the data for

    Returns
    -------
    The data for the unit
    """
    return _UNIT_DATA[unit]


def unit_race(unit_type: GameUnit) -> Union[units.Terran, units.Protoss, units.Zerg]:
    """Returns the race for the input unit

    Parameters
    ----------
    unit_type : unit to check the race of

    Returns
    -------
    Race of the unit
    """
    for race in [units.Terran, units.Protoss, units.Zerg]:
        if unit_type in list(race):
            return race


class GeneralUniformArmySelector(object):
    def __init__(
        self,
        available_units: List[GameUnit],
        minerals: int = np.Infinity,
        gas: int = np.Infinity,
        supply: int = np.Infinity,
        units: int = np.Infinity,
        seed: int = None,
    ):
        """
        Samples StarCraft II armies uniformly at random given a collection of available
        units, budgets on the armies (minerals, gas, supply, etc.), and bounds on where to
        spawn the units.
        """
        self.available_units = available_units
        self.minerals = minerals
        self.gas = gas
        self.supply = supply
        self.units = units
        self.rng = _get_rng(seed)

        # Determine the actual constraints that we're working with.
        self._constraints = build_constraints(
            self.available_units,
            minerals=self.minerals,
            gas=self.gas,
            supply=self.supply,
            units=self.units,
        )

        # Check that we have at least one valid constraint
        if not self._constraints:
            raise ValueError("Couldn't find any bounded constraints.")

        # Create the GeneralUniformSelector object
        self._gus = GeneralUniformSelector(self._constraints, seed=self.rng)

    def select(self):
        # Turn this sample into the right format
        sample = self._gus.select()
        if sample is None:
            return {}

        units = []
        for unit_ix, unit_type in enumerate(self.available_units):
            units.extend([unit_type for _ in range(sample[unit_ix])])

        return [{"unit_type": unit} for unit in units]


def build_constraints(
    available_units, minerals=np.Infinity, gas=np.Infinity, supply=np.Infinity, units=np.Infinity
):
    # Ignore constraints with infinite maximum. First, work with field constraints. Can't use a list
    # comprehension here because we're using locals().
    ret = []
    for field in ["minerals", "gas", "supply"]:
        if locals()[field] < np.Infinity:
            ret.append(
                ([getattr(unit_data(unit), field) for unit in available_units], locals()[field],)
            )

    # Constrain the total number of units
    if units < np.Infinity:
        ret.append(([1 for unit in available_units], units))

    return ret

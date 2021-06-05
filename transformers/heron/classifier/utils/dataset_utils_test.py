from typing import List

import numpy as np
import pytest
from adept.utils.util import DotDict
from pysc2.lib import units
from pysc2.lib.buffs import Buffs
from pysc2.lib.features import PlayerRelative
from pysc2.lib.named_array import NamedNumpyArray

from gamebreaker.data.unit_stats import BASE_STATS
from gamebreaker.data.unit_stats import get_stats
from gamebreaker.data.unit_stats import MAX_STATS
from gamebreaker.data.unit_stats import stats_to_ndarray
from gamebreaker.env.base.obs_idx import ContextObsIdx
from gamebreaker.unit_data import _AVAILABLE_UNITS


@pytest.mark.parametrize("stats", [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
def test_stats_to_ndarray(stats: List[int]):
    """
    Test for stats_to_ndarray

    Parameters
    ----------
    stats: List[int]
        The fake stats to pass into the function

    Returns
    -------
    None
    """
    stat_names = [
        "supply",
        "minerals",
        "gas",
        "health",
        "shield",
        "speed",
        "g_range",
        "g_dps",
        "g_damage",
        "a_range",
        "a_dps",
        "a_damage",
    ]
    dotdict_stats = DotDict({key: stat for key, stat in zip(stat_names, stats)})

    np_stats = stats_to_ndarray(dotdict_stats)

    assert all(i == j for i, j in zip(np_stats, stats))


def test_read_episode():
    pass


def test_agent_string_to_int():
    pass


class TestGetStats:
    features = [
        "unit_type",
        "alliance",
        "buff_id_0",
        "buff_id_1",
        "attack_upgrade_level",
        "on_creep",
    ]

    def test_base_stats(self):
        for unit_type in _AVAILABLE_UNITS:
            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            unit_type,
                            PlayerRelative.SELF,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, np.zeros(len(list(ContextObsIdx))))

            for stat in unit_stats:
                assert (
                    unit_stats[stat] == BASE_STATS[unit_type][stat] / MAX_STATS[stat]
                ), (
                    f"{unit_type.name}'s {stat}: Expected "
                    + f"{BASE_STATS[unit_type][stat] / MAX_STATS[stat]} got {unit_stats[stat]}"
                )

                assert (
                    unit_stats[stat] <= 1
                ), f"{unit_type.name} {stat}: {unit_stats[stat]} > 1"

    def test_creep_change(self):
        zerg_units = [unit_type for unit_type in BASE_STATS if unit_type in units.Zerg]

        for unit_type in zerg_units:
            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            unit_type,
                            PlayerRelative.SELF,
                            0,
                            0,
                            0,
                            1,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, np.zeros(len(list(ContextObsIdx))))

            # Every zerg unit (that we use) gains speed from creep
            assert unit_stats.speed > BASE_STATS[unit_type].speed / MAX_STATS.speed

    def test_zealot_charges(self):
        unit = NamedNumpyArray(
            np.asarray(
                [
                    [
                        units.Protoss.Zealot,
                        PlayerRelative.SELF,
                        0,
                        0,
                        0,
                        0,
                    ]
                ]
            ),
            (None, self.features),
        )[0]
        unit_stats = get_stats(unit, np.zeros(len(list(ContextObsIdx))))

        assert (
            unit_stats.speed == BASE_STATS[units.Protoss.Zealot].speed / MAX_STATS.speed
        )

    def test_adept_glaives(self):
        # Check that g_dps changes when it should
        for alliance, upgrade in zip(
            [PlayerRelative.SELF, PlayerRelative.ENEMY],
            [
                ContextObsIdx.blue_resonating_glaives,
                ContextObsIdx.red_resonating_glaives,
            ],
        ):
            upgrades = np.zeros(len(list(ContextObsIdx)))
            upgrades[upgrade] += 1

            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            units.Protoss.Adept,
                            alliance,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, upgrades)

            assert unit_stats.g_dps == 9 / MAX_STATS.g_dps

        # Check that it doesn't change when it shouldn't
        for alliance, upgrade in zip(
            [PlayerRelative.ENEMY, PlayerRelative.SELF],
            [
                ContextObsIdx.blue_resonating_glaives,
                ContextObsIdx.red_resonating_glaives,
            ],
        ):
            upgrades = np.zeros(len(list(ContextObsIdx)))
            upgrades[upgrade] += 1

            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            units.Protoss.Adept,
                            alliance,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, upgrades)

            assert (
                unit_stats.g_dps
                == BASE_STATS[units.Protoss.Adept].g_dps / MAX_STATS.g_dps
            )

    def test_marine_stims(self):
        # Check that g_dps changes when it should
        for buff_id_0, buff_id_1 in [
            (0, Buffs.Stimpack),
            (Buffs.Stimpack, 0),
            (Buffs.Stimpack, Buffs.Stimpack),
        ]:
            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            units.Terran.Marine,
                            PlayerRelative.SELF,
                            buff_id_0,
                            buff_id_1,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, np.zeros(len(list(ContextObsIdx))))

            assert unit_stats.g_dps == 14.7 / MAX_STATS.g_dps
            assert unit_stats.a_dps == 14.7 / MAX_STATS.a_dps
            assert (
                unit_stats.speed
                == (BASE_STATS[units.Terran.Marine].speed + 1.57) / MAX_STATS.speed
            )

    def test_marauder_stims(self):
        # Check that g_dps changes when it should
        for buff_id_0, buff_id_1 in [
            (0, Buffs.Stimpack),
            (Buffs.Stimpack, 0),
            (Buffs.Stimpack, Buffs.Stimpack),
        ]:
            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            units.Terran.Marauder,
                            PlayerRelative.SELF,
                            buff_id_0,
                            buff_id_1,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, np.zeros(len(list(ContextObsIdx))))

            assert unit_stats.g_dps == 14.1 / MAX_STATS.g_dps
            assert unit_stats.a_dps == 14.1 / MAX_STATS.a_dps
            assert (
                unit_stats.speed
                == (BASE_STATS[units.Terran.Marauder].speed + 1.57) / MAX_STATS.speed
            )

    def test_hellbat_preigniter(self):
        # Check that g_dps changes when it should
        for alliance, upgrade in zip(
            [PlayerRelative.SELF, PlayerRelative.ENEMY],
            [
                ContextObsIdx.blue_infernal_preigniter,
                ContextObsIdx.red_infernal_preigniter,
            ],
        ):
            upgrades = np.zeros(len(list(ContextObsIdx)))
            upgrades[upgrade] += 1

            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            units.Terran.Hellbat,
                            alliance,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, upgrades)

            assert (
                unit_stats.g_damage
                == (BASE_STATS[units.Terran.Hellbat].g_damage + 12) / MAX_STATS.g_damage
            )
            assert (
                unit_stats.g_dps
                == (BASE_STATS[units.Terran.Hellbat].g_dps + 8.4) / MAX_STATS.g_dps
            )

        # Check that it doesn't change when it shouldn't
        for alliance, upgrade in zip(
            [PlayerRelative.ENEMY, PlayerRelative.SELF],
            [
                ContextObsIdx.blue_infernal_preigniter,
                ContextObsIdx.red_infernal_preigniter,
            ],
        ):
            upgrades = np.zeros(len(list(ContextObsIdx)))
            upgrades[upgrade] += 1

            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            units.Terran.Hellbat,
                            alliance,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, upgrades)

            assert (
                unit_stats.g_damage
                == (BASE_STATS[units.Terran.Hellbat].g_damage) / MAX_STATS.g_damage
            )
            assert (
                unit_stats.g_dps
                == (BASE_STATS[units.Terran.Hellbat].g_dps) / MAX_STATS.g_dps
            )

    def test_zergling_glands(self):
        # Check that g_dps changes when it should
        for alliance, upgrade in zip(
            [PlayerRelative.SELF, PlayerRelative.ENEMY],
            [ContextObsIdx.blue_adrenal_glands, ContextObsIdx.red_adrenal_glands],
        ):
            upgrades = np.zeros(len(list(ContextObsIdx)))
            upgrades[upgrade] += 1

            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            units.Zerg.Zergling,
                            alliance,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, upgrades)

            assert unit_stats.g_dps == 14.3 / MAX_STATS.g_dps

        # Check that it doesn't change when it shouldn't
        for alliance, upgrade in zip(
            [PlayerRelative.ENEMY, PlayerRelative.SELF],
            [ContextObsIdx.blue_adrenal_glands, ContextObsIdx.red_adrenal_glands],
        ):
            upgrades = np.zeros(len(list(ContextObsIdx)))
            upgrades[upgrade] += 1

            unit = NamedNumpyArray(
                np.asarray(
                    [
                        [
                            units.Zerg.Zergling,
                            alliance,
                            0,
                            0,
                            0,
                            0,
                        ]
                    ]
                ),
                (None, self.features),
            )[0]
            unit_stats = get_stats(unit, upgrades)

            assert (
                unit_stats.g_dps
                == (BASE_STATS[units.Zerg.Zergling].g_dps) / MAX_STATS.g_dps
            )


def test_read_files():
    pass


def test_shuffle_and_save():
    pass


def test_determine_attrition():
    pass


def test_shuffle_dataset():
    pass

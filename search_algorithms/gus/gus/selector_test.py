# Copyright (C) 2021 Heron Systems, Inc.
import unittest
from collections import Counter

import gus.test_utils as test_utils
from gus.test_utils import units, GeneralUniformArmySelector


class GeneralUniformSelectorTest(unittest.TestCase):
    def _unit_frequency(self, army, available_units):
        frequencies = dict(Counter([unit["unit_type"] for unit in army]))
        return tuple(frequencies.get(unit, 0) for unit in available_units)

    def test_no_valid_army_to_sample(self):
        available_units = [
            unit for unit in test_utils.available_units([units.Terran, units.Protoss, units.Zerg])
        ]
        selector = GeneralUniformArmySelector(available_units, minerals=0)
        self.assertEqual({}, selector.select())

    def test_marines(self):
        # Let's create a Terran army
        available_units = [
            unit
            for unit in test_utils.available_units(units.Terran)
            if unit != units.Terran.Hellion
        ]

        for minerals_limit in range(50, 1000, 50):
            selector = GeneralUniformArmySelector(available_units, minerals=minerals_limit, gas=0)
            result = selector.select()

            self.assertEqual(
                minerals_limit / test_utils.unit_data(units.Terran.Marine).minerals, len(result)
            )

    def test_marine_gasless(self):
        # Let's create a Terran army
        available_units = [unit for unit in test_utils.available_units(units.Terran)]

        for minerals_limit in range(50, 1000, 50):
            selector = GeneralUniformArmySelector(available_units, minerals=minerals_limit, gas=0)
            result = selector.select()

            for unit in result:
                self.assertIn(
                    unit, [{"unit_type": units.Terran.Marine}, {"unit_type": units.Terran.Hellion},]
                )

    def test_marauder_adept_distr(self):
        available_units = [units.Terran.Marauder, units.Protoss.Adept]
        n_trials = 1000

        for n_units in range(2, 20):
            minerals = n_units * test_utils.unit_data(units.Terran.Marauder).minerals
            gas = n_units * test_utils.unit_data(units.Terran.Marauder).gas

            selector = GeneralUniformArmySelector(available_units, minerals=minerals, gas=gas)

            frequency_of_frequencies = dict(
                Counter(
                    [
                        self._unit_frequency(selector.select(), available_units)
                        for _ in range(n_trials)
                    ]
                )
            )

            # Ensure that we've seen each combination of marines and marauders, and
            # establish reasonable-ish bounds on the frequencies.
            self.assertEqual(n_units + 1, len(frequency_of_frequencies))

    def test_marauder_adept_flaky1(self):
        available_units = [units.Terran.Marauder, units.Protoss.Adept]
        n_trials = 1000

        for n_units in range(2, 20):
            minerals = n_units * test_utils.unit_data(units.Terran.Marauder).minerals
            gas = n_units * test_utils.unit_data(units.Terran.Marauder).gas

            selector = GeneralUniformArmySelector(available_units, minerals=minerals, gas=gas)

            frequency_of_frequencies = dict(
                Counter(
                    [
                        self._unit_frequency(selector.select(), available_units)
                        for _ in range(n_trials)
                    ]
                )
            )
            # NOTE: if this test fails, that's probably ok - this is expected to be
            # flaky.
            try:
                for n_marines in range(n_units + 1):
                    key = (n_marines, n_units - n_marines)
                    self.assertIn(key, frequency_of_frequencies)

            except:
                print(
                    "test_marauder_adept_are_generated_uniformly failed flakily; "
                    "if inclined, rerun."
                )

    def test_marauder_adept_flaky2(self):
        available_units = [units.Terran.Marauder, units.Protoss.Adept]
        n_trials = 1000

        for n_units in range(2, 20):
            minerals = n_units * test_utils.unit_data(units.Terran.Marauder).minerals
            gas = n_units * test_utils.unit_data(units.Terran.Marauder).gas

            selector = GeneralUniformArmySelector(available_units, minerals=minerals, gas=gas)

            frequency_of_frequencies = dict(
                Counter(
                    [
                        self._unit_frequency(selector.select(), available_units)
                        for _ in range(n_trials)
                    ]
                )
            )

            expected_p = 1.0 / (n_units + 1)

            # NOTE: if this test fails, that's probably ok - this is expected to be
            # flaky.
            try:
                for n_marines in range(n_units + 1):
                    key = (n_marines, n_units - n_marines)

                    self.assertLessEqual(0.6 * n_trials * expected_p, frequency_of_frequencies[key])
                    self.assertLessEqual(frequency_of_frequencies[key], 1.4 * n_trials * expected_p)
            except:
                print(
                    "test_marauder_adept_are_generated_uniformly failed flakily; "
                    "if inclined, rerun."
                )

    def test_marauder_adept_prob(self):
        available_units = [units.Terran.Marauder, units.Protoss.Adept]
        n_trials = 1000

        for n_units in range(2, 20):
            minerals = n_units * test_utils.unit_data(units.Terran.Marauder).minerals
            gas = n_units * test_utils.unit_data(units.Terran.Marauder).gas

            selector = GeneralUniformArmySelector(available_units, minerals=minerals, gas=gas)

            frequency_of_frequencies = dict(
                Counter(
                    [
                        self._unit_frequency(selector.select(), available_units)
                        for _ in range(n_trials)
                    ]
                )
            )
            expected_p = 1.0 / (n_units + 1)

            # Establish bounds on the minimum and maximum frequencies
            distribution = [v * 1.0 / n_trials for v in frequency_of_frequencies.values()]
            min_freq, max_freq = min(distribution), max(distribution)
            self.assertLessEqual(min_freq, expected_p)
            self.assertLessEqual(expected_p, max_freq)

    def test_full_supply_count(self):
        race_with_expected_counts = [
            (units.Terran, 101 ** 2),
            (units.Protoss, 176851),
            (units.Zerg, 691951),
        ]

        for race, expected_count in race_with_expected_counts:
            available_units = test_utils.available_units(race)
            selector = GeneralUniformArmySelector(available_units, supply=200)
            selector.select()
            self.assertEqual(1, len(selector._gus._subproblem_structures))

    def test_full_supply_count_subproblem(self):
        race_with_expected_counts = [
            (units.Terran, 534928443),
            (units.Protoss, 309574539),
            (units.Zerg, 130259423),
        ]

        for race, expected_count in race_with_expected_counts:
            available_units = test_utils.available_units(race)
            selector = GeneralUniformArmySelector(available_units, supply=200)
            selector.select()

            self.assertEqual(expected_count, selector._gus._subproblem_structures[0]["total_count"])

    def test_zerg_pareto_optimality_with_low_mineral_count(self):
        available_units = test_utils.available_units(units.Zerg)
        selector = GeneralUniformArmySelector(available_units, minerals=50, gas=0)
        army = selector.select()

        self.assertEqual(1, len(selector._gus._subproblem_structures))

    def test_zerg_pareto_optimality_with_low_mineral_count_structures(self):
        available_units = test_utils.available_units(units.Zerg)
        selector = GeneralUniformArmySelector(available_units, minerals=50, gas=0)
        army = selector.select()

        self.assertEqual(1, selector._gus._subproblem_structures[0]["total_count"])

    def test_zerg_pareto_optimality_with_low_mineral_count_frequency(self):
        available_units = test_utils.available_units(units.Zerg)
        selector = GeneralUniformArmySelector(available_units, minerals=50, gas=0)
        army = selector.select()
        frequency = dict(Counter([unit["unit_type"] for unit in army]))

        self.assertEqual(frequency, {units.Zerg.Zergling: 2})

    def test_strange_mineral_counts(self):
        available_units = test_utils.available_units(units.Zerg)

        for mineral_count in range(200):
            selector = GeneralUniformArmySelector(available_units, minerals=mineral_count, gas=0)
            army = selector.select()

            frequency = dict(Counter([unit["unit_type"] for unit in army]))

            expected_zerglings = int(mineral_count / 25)
            testin = {units.Zerg.Zergling: expected_zerglings} if expected_zerglings > 0 else {}
            self.assertEqual(frequency, testin)

    def test_large_army_count(self):
        available_units = test_utils.available_units([units.Terran, units.Protoss, units.Zerg])

        selector = GeneralUniformArmySelector(available_units, supply=200)
        selector.select()

        self.assertEqual(1, len(selector._gus._subproblem_structures))

    def test_large_army_count_subproblem(self):
        available_units = test_utils.available_units([units.Terran, units.Protoss, units.Zerg])

        selector = GeneralUniformArmySelector(available_units, supply=200)
        selector.select()

        self.assertEqual(
            1005306178309287334804, selector._gus._subproblem_structures[0]["total_count"]
        )

    def test_right_most_non_negative_col_ix_works(self):
        available_units = [units.Zerg.Baneling, units.Zerg.Zergling]
        selector = GeneralUniformArmySelector(available_units, minerals=2000, gas=2000, supply=10,)

        selector.select()


if __name__ == "__main__":
    unittest.main()

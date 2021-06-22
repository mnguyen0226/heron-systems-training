// Copyright (C) 2021 Heron Systems, Inc.
#include "gus/selector.h"

#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <unordered_map>

#include "gus/selector_test.h"
#include "uint128_t/uint128_t.h"

std::vector<Unit> ScUnitUtils::units() {
    std::vector<Unit> units;
    for (const auto& kv : ScUnitUtils::all_units) {
        units.push_back(kv.second);
    }
    return units;
}

std::vector<Unit> ScUnitUtils::available_units(const std::vector<Race>& races) {
    std::vector<Unit> ret;
    for (const auto& kv : ScUnitUtils::all_units) {
        if (std::find(races.begin(), races.end(), kv.second.race) !=
            races.end()) {
            ret.push_back(kv.second);
        }
    }
    return ret;
}

const Unit& ScUnitUtils::unit_data(const std::string& unit) {
    return all_units.at(unit);
}

std::vector<Unit> ScUnitUtils::str_to_units(
    const std::vector<std::string>& available_units) {
    std::vector<Unit> units_list;
    for (const auto& unit : available_units) {
        units_list.push_back(ScUnitUtils::unit_data(unit));
    }
    return units_list;
}

TEST(SelectorTest, CreateADollar) {
    Eigen::MatrixXf constraints(1, 7);
    constraints << 0.01, 0.05, 0.10, 0.25, 0.50, 1, 1;

    GeneralUniformSelector<uint64_t, int> gus(constraints);
    ASSERT_EQ(1, gus.counts().size());
    ASSERT_EQ(293, gus.counts()[0]);
}

TEST(SelectorTest, MarauderAdept) {
    BasicGeneralUniformArmySelector guas(
        std::vector<std::string>{"Adept", "Marauder"}, 1000, 250, 20, 10);
    const auto& gus = guas.gus();

    ASSERT_EQ(4, gus.counts().size());

    for (size_t i = 0; i < gus.counts().size(); ++i) {
        ASSERT_EQ(11, gus.counts()[i]);
    }
}

TEST(SelectorTest, NoValidArmyToSample) {
    BasicGeneralUniformArmySelector gaus(ScUnitUtils::available_units(), 0);
    ASSERT_EQ(0, gaus.select().maxCoeff());
}

TEST(SelectorTest, TestMarines) {
    // Let's create a Terran army
    std::vector<Unit> available_units;
    for (const auto& unit : ScUnitUtils::available_units({Race::Terran})) {
        if (unit.name != std::string("Hellion")) {
            available_units.push_back(unit);
        }
    }

    for (int minerals_limit = 50; minerals_limit < 1000; minerals_limit += 50) {
        BasicGeneralUniformArmySelector gaus(available_units, minerals_limit,
                                             0);
        auto army = gaus.select_sparse();

        ASSERT_EQ(minerals_limit / ScUnitUtils::unit_data("Marine").minerals,
                  army["Marine"]);
    }
}

TEST(SelectorTest, TestMarineGasless) {
    // Let's create a Terran army
    for (int minerals_limit = 50; minerals_limit < 1000; minerals_limit += 50) {
        BasicGeneralUniformArmySelector gaus(
            ScUnitUtils::available_units({Race::Terran}), minerals_limit, 0);
        auto army = gaus.select_sparse();

        for (const auto& kv : army) {
            ASSERT_TRUE(kv.first == std::string("Marine") ||
                        kv.first == std::string("Hellion"));
        }
    }
}

TEST(SelectorTest, TestFullSupplyCount) {
    std::unordered_map<Race, uint64_t> counts({{Race::Terran, 534928443},
                                               {Race::Protoss, 309574539},
                                               {Race::Zerg, 130259423}});

    for (const auto& kv : counts) {
        BasicGeneralUniformArmySelector gaus(
            ScUnitUtils::available_units({kv.first}), -1, -1, 200);
        ASSERT_EQ(1, gaus.gus().counts().size());
        ASSERT_EQ(kv.second, gaus.gus().counts()[0]);
    }
}

TEST(SelectorTest, ZergParetoOptimalityWithLowMineralCount) {
    BasicGeneralUniformArmySelector gaus(
        ScUnitUtils::available_units({Race::Zerg}), 50, 0);
    ASSERT_EQ(1, gaus.gus().counts().size());
    ASSERT_EQ(1, gaus.gus().counts()[0]);

    auto actual = gaus.select_sparse();
    std::unordered_map<std::string, size_t> expected({{"Zergling", 2}});

    ASSERT_EQ(actual, expected);
}

TEST(SelectorTest, TestStrangeMineralCounts) {
    for (size_t mineral_count = 0; mineral_count < 200; ++mineral_count) {
        BasicGeneralUniformArmySelector gaus(
            ScUnitUtils::available_units({Race::Zerg}), mineral_count, 0);
        auto actual = gaus.select_sparse();

        std::unordered_map<std::string, size_t> expected;
        size_t expected_zerglings = mineral_count / 25;
        if (expected_zerglings > 0) {
            expected["Zergling"] = expected_zerglings;
        }

        ASSERT_EQ(expected, actual);
    }
}

TEST(SelectorTest, LargeArmyCount) {
    BigGeneralUniformArmySelector gaus(
        ScUnitUtils::available_units({Race::Terran, Race::Protoss, Race::Zerg}),
        -1, -1, 200);
    ASSERT_EQ(1, gaus.gus().counts().size());
    ASSERT_EQ(std::string("1005306178309287334804"),
              gaus.gus().counts()[0].str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
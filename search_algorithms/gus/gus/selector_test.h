// Copyright (C) 2021 Heron Systems, Inc.
#include <Eigen/Eigen>
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gus/selector.h"
#include "uint128_t/uint128_t.h"

enum Race { Terran, Protoss, Zerg };

struct Unit {
    std::string name;
    Race race;
    float supply;
    uint16_t minerals;
    uint16_t gas;
    uint16_t health;
    uint16_t shield;
    float range;
};

class ScUnitUtils {
  protected:
    static std::unordered_map<std::string, Unit> all_units;

  public:
    static std::vector<Unit> units();
    static std::vector<Unit> available_units(const std::vector<Race>& races = {
                                                 Race::Protoss, Race::Terran,
                                                 Race::Zerg});
    static const Unit& unit_data(const std::string& unit);
    static std::vector<Unit> str_to_units(
        const std::vector<std::string>& available_units);
};

std::unordered_map<std::string, Unit> ScUnitUtils::all_units{[] {
    std::unordered_map<std::string, Unit> ret;
    std::vector<Unit> units_vec = {
        // Protoss units
        {"Zealot", Race::Protoss, 2, 100, 0, 100, 50, 0.1},
        {"Sentry", Race::Protoss, 2, 50, 100, 40, 40, 5},
        {"Stalker", Race::Protoss, 2, 125, 50, 80, 80, 6},
        {"Adept", Race::Protoss, 2, 100, 25, 70, 70, 4},
        {"Immortal", Race::Protoss, 4, 275, 100, 200, 100, 6},
        {"Colossus", Race::Protoss, 6, 300, 200, 200, 150, 7},
        {"HighTemplar", Race::Protoss, 2, 50, 150, 40, 40, 6},
        // Terran units
        {"Marine", Race::Terran, 1, 50, 0, 45, 0, 5},
        {"Marauder", Race::Terran, 2, 100, 25, 125, 0, 6},
        {"Reaper", Race::Terran, 1, 50, 50, 60, 0, 5},
        {"SiegeTank", Race::Terran, 3, 150, 125, 175, 0, 7},
        {"Hellion", Race::Terran, 2, 100, 0, 90, 0, 5},
        {"Cyclone", Race::Terran, 3, 150, 100, 120, 0, 5},
        {"Thor", Race::Terran, 6, 300, 200, 400, 0, 10},
        // Zerg units
        {"Zergling", Race::Zerg, 0.5, 25, 0, 35, 0, 0.1},
        {"Baneling", Race::Zerg, 0.5, 25, 25, 30, 0, 2.2},
        {"Roach", Race::Zerg, 2, 75, 25, 145, 0, 4},
        {"Hydralisk", Race::Zerg, 2, 100, 50, 90, 0, 5},
        {"Ultralisk", Race::Zerg, 6, 300, 200, 500, 0, 1},
        {"Infestor", Race::Zerg, 2, 100, 150, 90, 0, 0}};

    for (const auto& unit : units_vec) {
        ret[unit.name] = unit;
    }

    return ret;
}()};

template <typename CountType, typename IntType>
class GeneralUniformArmySelector {
  public:
    GeneralUniformArmySelector(const std::vector<std::string> available_units,
                               int minerals = -1, int gas = -1, int supply = -1,
                               int units = -1, int seed = 1337,
                               uint64_t max_denominator = 100000)
        : GeneralUniformArmySelector(ScUnitUtils::str_to_units(available_units),
                                     minerals, gas, supply, units, seed,
                                     max_denominator) {}

    GeneralUniformArmySelector(const std::vector<Unit>& available_units,
                               int minerals = -1, int gas = -1, int supply = -1,
                               int units = -1, int seed = 1337,
                               uint64_t max_denominator = 100000)
        : available_units_(available_units),
          minerals_(minerals),
          gas_(gas),
          supply_(supply),
          units_(units),
          seed_(seed) {
        size_t n_constraints =
            (minerals >= 0) + (gas >= 0) + (supply >= 0) + (units >= 0);
        Eigen::MatrixXf constraints(n_constraints, available_units_.size() + 1);

        size_t row_ix = -1;
        if (minerals >= 0) {
            ++row_ix;
            constraints(row_ix, available_units.size()) = minerals;
            for (size_t unit_ix = 0; unit_ix < available_units_.size();
                 ++unit_ix) {
                constraints(row_ix, unit_ix) =
                    available_units_[unit_ix].minerals;
            }
        }

        if (gas >= 0) {
            ++row_ix;
            constraints(row_ix, available_units_.size()) = gas;
            for (size_t unit_ix = 0; unit_ix < available_units_.size();
                 ++unit_ix) {
                constraints(row_ix, unit_ix) = available_units_[unit_ix].gas;
            }
        }

        if (supply >= 0) {
            ++row_ix;
            constraints(row_ix, available_units_.size()) = supply;
            for (size_t unit_ix = 0; unit_ix < available_units_.size();
                 ++unit_ix) {
                constraints(row_ix, unit_ix) = available_units_[unit_ix].supply;
            }
        }

        if (units >= 0) {
            ++row_ix;
            constraints(row_ix, available_units_.size()) = units;
            for (size_t unit_ix = 0; unit_ix < available_units_.size();
                 ++unit_ix) {
                constraints(row_ix, unit_ix) = 1;
            }
        }

        gus_ = new GeneralUniformSelector<CountType, IntType>(constraints, seed,
                                                              max_denominator);
    }

    ~GeneralUniformArmySelector() { delete gus_; }

    const GeneralUniformSelector<CountType, IntType>& gus() const {
        return *gus_;
    }

    typename GeneralUniformSelector<CountType, IntType>::MatrixXui select() {
        return this->gus_->select();
    }

    std::unordered_map<std::string, size_t> select_sparse() {
        auto army = this->select();
        std::unordered_map<std::string, size_t> ret;
        for (size_t unit_ix = 0; unit_ix < available_units_.size(); ++unit_ix) {
            if (army(unit_ix)) {
                ret[available_units_[unit_ix].name] = army(unit_ix);
            }
        }
        return ret;
    }

  protected:
    std::vector<Unit> available_units_;
    int minerals_;
    int gas_;
    int supply_;
    int units_;
    int seed_;

    Eigen::MatrixXf constraints_;
    GeneralUniformSelector<CountType, IntType>* gus_;
};

using BasicGeneralUniformArmySelector =
    GeneralUniformArmySelector<uint32_t, int>;
using BigGeneralUniformArmySelector =
    GeneralUniformArmySelector<uint128_t, int>;
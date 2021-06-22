// Copyright (C) 2021 Heron Systems, Inc.
#pragma once

#include <random>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include "Eigen/Eigen"

namespace std {
template <typename Scalar, int Rows, int Cols>
struct hash<Eigen::Matrix<Scalar, Rows, Cols>> {
    // https://wjngkoh.wordpress.com/2015/03/04/c-hash-function-for-eigen-matrix-and-vector/
    size_t operator()(const Eigen::Matrix<Scalar, Rows, Cols>& matrix) const {
        size_t seed = 0;
        for (size_t i = 0; i < static_cast<size_t>(matrix.size()); ++i) {
            Scalar elem = *(matrix.data() + i);
            seed ^= std::hash<Scalar>()(elem) + 0x9e3779b9 + (seed << 6) +
                    (seed >> 2);
        }
        return seed;
    }
};
};  // namespace std

template <typename CountType, typename IntType>
class GeneralUniformSelector {
  public:
    using uIntType = typename std::make_unsigned<IntType>::type;
    using MatrixXi = Eigen::Matrix<IntType, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXui = Eigen::Matrix<uIntType, Eigen::Dynamic, Eigen::Dynamic>;

    GeneralUniformSelector(const Eigen::MatrixXf& constraints,
                           const int seed = 0,
                           const uint64_t max_denominator = 100000ULL);

    MatrixXui select();

    const std::vector<CountType>& counts() const { return counts_; }
    const Eigen::MatrixXf& constraints() const { return constraints_; }

  protected:
    MatrixXui create_single_subproblem(const MatrixXui& constraints);
    void generate_subproblems();
    void construct_structures();
    CountType generate_structure(size_t subproblem_ix, MatrixXi b,
                                 size_t col_ix = 0);
    void reformat_constraint(size_t constraint_ix);
    std::vector<uIntType> tighten_constraint(size_t constraint_ix);
    MatrixXui sample(size_t subproblem_ix);

    Eigen::MatrixXf constraints_;
    MatrixXui clean_constraints_;
    int seed_;
    uint64_t max_denominator_;

    std::mt19937 rng_;
    size_t n_constraints_;
    size_t n_vars_;
    std::vector<MatrixXui, Eigen::aligned_allocator<MatrixXui>> subproblems_;

    std::vector<std::unordered_map<
        MatrixXi, std::tuple<CountType, std::vector<CountType>>>>
        subproblem_structures_;

    std::vector<CountType> counts_;
};

#include "gus/selector_impl.cc"

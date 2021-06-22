// Copyright (C) 2021 Heron Systems, Inc.
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "gus/selector.h"

template <typename T>
T gcd(T n1, T n2) {
    while (n2) std::swap(n1 %= n2, n2);
    return n1;
}

template <typename CountType>
size_t get_sample(const std::vector<CountType>& frequencies, std::mt19937& rng,
                  CountType count = 0) {
    if (count <= 0) {
        count = std::accumulate(frequencies.begin(), frequencies.end(), 0);
    }

    std::uniform_int_distribution<CountType> dist(1, count);
    CountType rn = dist(rng);
    size_t sample = -1;
    for (size_t i = 0; i < frequencies.size(); ++i) {
        if (rn <= frequencies[i]) {
            sample = i;
            break;
        }
        rn -= frequencies[i];
    }
    return sample;
}

std::tuple<uint64_t, uint64_t> as_integer_ratio(double value,
                                                uint64_t max_denominator) {
    // Closest fraction with denominator at most max_denominator
    // Taken from
    // https://github.com/python/cpython/blob/v3.7.1/Objects/floatobject.c#L1534L1604
    int exponent;
    double float_part = frexp(value, &exponent);

    for (int i = 0; i < 300 && float_part != std::floor(float_part); i++) {
        float_part *= 2.0;
        exponent--;
    }

    long n = float_part;
    uint64_t d = 1;

    if (exponent > 0) {
        n <<= exponent;
    } else {
        d <<= exponent;
    }

    d = 1ULL << (-exponent);

    // Taken from limit_denominator under /usr/lib/python3.7/fractions.py.
    if (d <= max_denominator) {
        return std::make_tuple((uint64_t)n, (uint64_t)d);
    }

    long p0 = 0, q0 = 1, p1 = 1, q1 = 0;

    while (true) {
        long a = n / d;
        long q2 = q0 + a * q1;
        if (q2 > max_denominator) {
            break;
        }
        std::tie(p0, q0, p1, q1) = std::make_tuple(p1, q1, p0 + a * p1, q2);
        std::tie(n, d) = std::make_tuple(d, n - a * d);
    }

    long k = (max_denominator - q0) / q1;
    double bound_1 = (p0 + k * p1) / (q0 + k * q1);
    double bound_2 = p1 / q1;
    double cur = n / d;

    return std::abs(bound_2 - cur) <= std::abs(bound_1 - cur)
               ? std::make_tuple((uint64_t)p1, (uint64_t)q1)
               : std::make_tuple((uint64_t)(p0 + k * p1),
                                 (uint64_t)(q0 + k * q1));
}

template <typename CountType, typename IntType>
GeneralUniformSelector<CountType, IntType>::GeneralUniformSelector(
    const Eigen::MatrixXf& constraints, const int seed,
    const uint64_t max_denominator)
    : constraints_(constraints),
      seed_(seed),
      max_denominator_(max_denominator) {
    rng_ = std::mt19937(seed_);
    n_constraints_ = constraints.rows();
    n_vars_ = constraints.cols() - 1;
    clean_constraints_ =
        GeneralUniformSelector<CountType, IntType>::MatrixXui::Zero(
            constraints.rows(), constraints.cols());

    // This is almost correct; there are very tiny oversampling issues when
    // multiple constraints are simultaneously tight. We need to use the
    // principle of inclusion exclusion to address this, but oh well. These
    // oversampling issues typically end up being very small.
    this->construct_structures();
}

template <typename CountType, typename IntType>
void GeneralUniformSelector<CountType, IntType>::reformat_constraint(
    size_t constraint_ix) {
    // Determine the least number to multiply each side of the constraint by to
    // turn all coefficients into integers.
    std::vector<uint64_t> denoms(constraints_.cols());
    uint64_t lcm = 1ULL;
    for (size_t i = 0; i < denoms.size(); ++i) {
        auto coeff =
            as_integer_ratio(constraints_(constraint_ix, i), max_denominator_);
        lcm = std::lcm(lcm, std::get<1>(coeff));
    }

    // After multiplying by the lcm and rounding, we'll have a cleaned integer
    // matrix.
    clean_constraints_.row(constraint_ix) =
        (constraints_.row(constraint_ix) * lcm)
            .array()
            .round()
            .matrix()
            .cast<GeneralUniformSelector<CountType, IntType>::uIntType>();

    // Find a common divisor if this exists to keep the numbers low
    auto cur_gcd = clean_constraints_(constraint_ix, 0);
    for (size_t i = 1; i < clean_constraints_.cols(); ++i) {
        cur_gcd = gcd(cur_gcd, clean_constraints_(constraint_ix, i));
    }

    clean_constraints_.row(constraint_ix) =
        clean_constraints_.row(constraint_ix) / cur_gcd;
}

template <typename CountType, typename IntType>
typename GeneralUniformSelector<CountType, IntType>::MatrixXui
GeneralUniformSelector<CountType, IntType>::create_single_subproblem(
    const GeneralUniformSelector<CountType, IntType>::MatrixXui& constraints) {
    // Attach slack variables
    using MatrixXui = GeneralUniformSelector<CountType, IntType>::MatrixXui;
    MatrixXui subproblem(n_constraints_, n_vars_ + n_constraints_);
    subproblem << constraints.leftCols(n_vars_),
        MatrixXui::Zero(n_constraints_, n_constraints_ - 1),
        constraints.rightCols(1);

    subproblem.block(1, n_vars_, n_constraints_ - 1, n_constraints_ - 1) =
        MatrixXui::Identity(n_constraints_ - 1, n_constraints_ - 1);

    // Detect whether we can create infinitely many units
    auto csum = subproblem.leftCols(n_vars_).colwise().sum();
    for (size_t ix = 0; ix < csum.size(); ++ix) {
        if (csum(ix) == 0) {
            std::stringstream ss;
            ss << "Unit " << ix
               << " can be produced infinitely given the constraints.";
            throw std::runtime_error(ss.str().c_str());
        }
    }

    return subproblem;
}

template <typename CountType, typename IntType>
void GeneralUniformSelector<CountType, IntType>::generate_subproblems() {
    using uIntType = GeneralUniformSelector<CountType, IntType>::uIntType;

    // Eliminate constraints with only zeros in the lhs and reformat those that
    // pass.
    for (size_t constraint_ix = 0; constraint_ix < n_constraints_;
         ++constraint_ix) {
        // Check that all coefficients are non-negative
        if (constraints_.row(constraint_ix).leftCols(n_vars_).minCoeff() < 0) {
            throw std::invalid_argument(
                (std::string("Found a negative coefficient in row ") +
                 std::to_string(constraint_ix) +
                 std::string(" of the constraints matrix."))
                    .c_str());
        }

        // Reformat the current constraint
        this->reformat_constraint(constraint_ix);
    }

    for (size_t constraint_ix = 0; constraint_ix < n_constraints_;
         ++constraint_ix) {
        // Tighten the current constraint
        std::vector<uIntType> tight_rhss =
            this->tighten_constraint(constraint_ix);

        // Create subproblem
        for (uIntType tight_rhs : tight_rhss) {
            // Create a constraints matrix where we set the first constraint to
            // be the tight one
            auto cur_constraints = clean_constraints_;
            cur_constraints(constraint_ix, n_vars_) = tight_rhs;
            cur_constraints.row(constraint_ix).swap(cur_constraints.row(0));

            // Turn this constraints matrix into a subproblem
            auto subproblem = this->create_single_subproblem(cur_constraints);

            // Eliminate subproblems where the right-hand side of the tight
            // constraint is 0; this can lead to generating solutions that are
            // not Pareto-optimal. E.g., if we have a mineral budget of 50 and a
            // gas budget of 0 and are trying to generate a Zerg army, in the
            // case where we attempt to enforce tightness on the gas budget and
            // keep slackness on the mineral budget, we would end up with 2
            // valid armies: (1) a single zergling, (2) two zerglings. Both
            // solutions would force tightness on the gas constraint and allow
            // for slackness on the mineral constraint. However, generating a
            // single zergling is not Pareto optimal.
            if (subproblem.template rightCols<1>()(0) > 0) {
                subproblems_.push_back(subproblem);
            }
        }
    }
}

template <typename CountType, typename IntType>
std::vector<typename GeneralUniformSelector<CountType, IntType>::uIntType>
GeneralUniformSelector<CountType, IntType>::tighten_constraint(
    size_t constraint_ix) {
    // Sometimes, we need to make constraint a bit loose, particularly if it
    // isn't possible to attain equality. E.g., if we have the equation
    //
    // 5x + 7y + 7z + 7w = 15
    //
    // (where = is tight-ish; meaning, we can't squeeze in another unit) we want
    // to expand the right hand side to include attainable values between 11 and
    // 15, inclusive; so, that's 14 and 15.
    using uIntType = GeneralUniformSelector<CountType, IntType>::uIntType;

    std::set<uIntType> attainable_numbers;
    uIntType rhs = clean_constraints_(constraint_ix, n_vars_);

    for (size_t ix = 0; ix < n_vars_; ++ix) {
        uIntType coeff = clean_constraints_(constraint_ix, ix);
        if (coeff > 0) {
            for (size_t j = 0; j < rhs + coeff; j += coeff) {
                attainable_numbers.insert(j);
            }
        }
    }

    // In the above example, we only want to keep values of the right hand side
    // for which it is impossible to squeeze in another unit. In the above case,
    // the value of the right hand side must be strictly greater than 15 - 5.
    uIntType min_positive_coeff = 0;
    for (size_t ix = 0; ix < n_vars_; ++ix) {
        uIntType cur_coeff = clean_constraints_(constraint_ix, ix);
        if (cur_coeff > 0) {
            min_positive_coeff = min_positive_coeff == 0
                                     ? cur_coeff
                                     : std::min(cur_coeff, min_positive_coeff);
        }
    }

    std::vector<uIntType> ret;
    for (const uIntType& e : attainable_numbers) {
        if (rhs < e + min_positive_coeff && e <= rhs) {
            ret.push_back(e);
        }
    }

    return ret;
}

template <typename CountType, typename IntType>
typename GeneralUniformSelector<CountType, IntType>::MatrixXi
create_structure_key(
    size_t col_ix,
    const typename GeneralUniformSelector<CountType, IntType>::MatrixXi& b) {
    typename GeneralUniformSelector<CountType, IntType>::MatrixXi ret(
        b.rows() + 1, 1);
    ret << col_ix, b;
    return ret;
}

template <typename CountType, typename IntType>
CountType GeneralUniformSelector<CountType, IntType>::generate_structure(
    size_t subproblem_ix,
    GeneralUniformSelector<CountType, IntType>::MatrixXi b, size_t col_ix) {
    // Only consider cases with positive b
    auto key = create_structure_key<CountType, IntType>(col_ix, b);
    if (b.minCoeff() < 0) {
        return 0;
    }

    const auto& A = subproblems_[subproblem_ix];
    size_t m = A.rows(), n = A.cols() - 1;
    size_t n_slacks = m - 1;
    size_t n_vars = n - n_slacks;
    auto& structure = subproblem_structures_[subproblem_ix];

    // When considering purely a subset of slack variables, we only ever have
    // one solution
    if (col_ix >= n_vars) {
        structure[key] = std::make_tuple(static_cast<CountType>(1),
                                         std::vector<CountType>());
        return static_cast<CountType>(1);
    }

    // When considering a single actual variable and a subset of slack
    // variables... find the right-most positive column index.
    size_t right_most_positive_col_ix = A.cols();
    for (size_t ix = n_vars - 1; ix >= 0; --ix) {
        if (A(0, ix) > 0) {
            right_most_positive_col_ix = ix;
            break;
        }
    }

    if (right_most_positive_col_ix == A.cols()) {
        throw std::runtime_error(
            "This should never happen. Complain to Karthik.");
    }

    using uIntType = GeneralUniformSelector<CountType, IntType>::uIntType;
    if (col_ix == right_most_positive_col_ix) {
        uIntType actual_var_value = b(0) / A(0, col_ix);

        // (1) check whether the coefficient of the actual variable divides b[0]
        if (b(0) % A(0, col_ix) != 0) {
            structure[key] = std::make_tuple(static_cast<CountType>(0),
                                             std::vector<CountType>());
            return static_cast<CountType>(0);
        }

        // (2) when substituted into the rest of the equations, check that the
        // slacks are all positive
        GeneralUniformSelector<CountType, IntType>::MatrixXi checks =
            b - (actual_var_value * A.col(col_ix)).template cast<IntType>();
        if (checks.minCoeff() < 0) {
            structure[key] = std::make_tuple(static_cast<CountType>(0),
                                             std::vector<CountType>());
            return static_cast<CountType>(0);
        }

        // If both checks pass, then we have a single solution
        structure[key] = std::make_tuple(static_cast<CountType>(1),
                                         std::vector<CountType>());
        return static_cast<CountType>(1);
    }

    // The recursive case
    std::vector<CountType> cases;
    auto cur_b = b;
    size_t t = 0;

    while (cur_b.minCoeff() >= 0) {
        key = create_structure_key<CountType, IntType>(col_ix + 1, cur_b);
        auto it = structure.find(key);

        CountType count =
            (it != structure.end())
                ? std::get<0>(it->second)
                : this->generate_structure(subproblem_ix, cur_b, col_ix + 1);

        cases.push_back(count);
        cur_b = b - (++t * A.col(col_ix)).template cast<IntType>();
    }

    CountType count =
        std::accumulate(cases.begin(), cases.end(), static_cast<CountType>(0));

    key = create_structure_key<CountType, IntType>(col_ix, b);
    structure[key] = std::make_tuple(count, cases);

    return count;
}

template <typename CountType, typename IntType>
void GeneralUniformSelector<CountType, IntType>::construct_structures() {
    this->generate_subproblems();
    counts_ = std::vector<CountType>(subproblems_.size());

    // Generate structures for each of the subproblems
    using MatrixXi = GeneralUniformSelector<CountType, IntType>::MatrixXi;
    using uIntType = GeneralUniformSelector<CountType, IntType>::uIntType;

    subproblem_structures_ = std::vector<std::unordered_map<
        MatrixXi, std::tuple<CountType, std::vector<CountType>>>>(
        subproblems_.size());

    for (size_t ix = 0; ix < subproblems_.size(); ++ix) {
        subproblem_structures_[ix] =
            std::unordered_map<MatrixXi,
                               std::tuple<CountType, std::vector<CountType>>>();
        counts_[ix] = this->generate_structure(
            ix, subproblems_[ix].rightCols(1).template cast<IntType>());
    }
}

template <typename CountType, typename IntType>
typename GeneralUniformSelector<CountType, IntType>::MatrixXui
GeneralUniformSelector<CountType, IntType>::sample(size_t subproblem_ix) {
    const auto& A = subproblems_[subproblem_ix];
    size_t m = A.rows(), n = A.cols() - 1;
    size_t n_slacks = m - 1;
    size_t n_vars = n - n_slacks;

    using MatrixXui = GeneralUniformSelector<CountType, IntType>::MatrixXui;
    using uIntType = GeneralUniformSelector<CountType, IntType>::uIntType;
    MatrixXui ret = MatrixXui::Zero(n, 1);
    MatrixXi b = A.rightCols(1).template cast<IntType>();
    auto orig_b = A.rightCols(1);
    auto& structure = subproblem_structures_[subproblem_ix];

    // Start with col_ix = 0 and the initial b
    for (size_t col_ix = 0; col_ix < n_vars - 1; ++col_ix) {
        auto key = create_structure_key<CountType, IntType>(col_ix, b);

        // Draw a sample for the current column
        CountType count;
        std::vector<CountType> frequencies;
        std::tie(count, frequencies) = structure.at(key);

        size_t sample = get_sample<CountType>(frequencies, this->rng_, count);
        ret(col_ix) = sample;

        // Update b
        b -= (A.col(col_ix) * sample).template cast<IntType>();
    }

    // Fill in the final variable
    ret(n_vars - 1) =
        (orig_b(0) -
         (A.row(0).leftCols(n_vars - 1) * ret.topRows(n_vars - 1)).value()) /
        A(0, n_vars - 1);

    // Fill in the final variable and slacks
    for (size_t i = 0; i < n_slacks; ++i) {
        ret(n_vars + i) = (orig_b(1) - (A.row(i + 1).leftCols(n_vars + i + 1) *
                                        ret.topRows(n_vars + i + 1))
                                           .value()) /
                          A(i + 1, n_vars + i);
    }

    return ret;
}

template <typename CountType, typename IntType>
typename GeneralUniformSelector<CountType, IntType>::MatrixXui
GeneralUniformSelector<CountType, IntType>::select() {
    // This is almost correct; there are very tiny oversampling issues when
    // multiple constraints are simultaneously tight. We need to use the
    // principle of inclusion exclusion to address this, but oh well. These
    // oversampling issues typically end up being very small.
    if (std::accumulate(counts_.begin(), counts_.end(), 0) == 0) {
        return GeneralUniformSelector<CountType, IntType>::MatrixXui::Zero(
            n_vars_, 1);
    }

    // Determine which substructure to sample from
    size_t substructure_ix = get_sample(counts_, this->rng_);
    return this->sample(substructure_ix);
}

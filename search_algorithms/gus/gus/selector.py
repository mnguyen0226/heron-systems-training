# Copyright (C) 2021 Heron Systems, Inc.
from fractions import Fraction
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from numpy.lib.financial import ipmt


def _get_rng(seed):
    if seed is None:
        return np.random
    elif type(seed) == int:
        return np.random.RandomState(seed=seed)
    else:
        # This happens when a user passes in a custom generator
        return seed


class GeneralUniformSelector(object):
    def __init__(
        self, constraints: List[Tuple[List, int]], seed: int = None,
    ):
        """
        Suppose that we are given n available items that we can pick from. Let x_j denote
        the number of unit j that we pick. Of course, x_j is non-negative and integral. A
        user may define m linear budget constraints which may be summarized as:

        sum_{j = 1} ^ {n} A_{ij} x_j <= b_{i}

        where i \in [1, 2, \cdots m]. How do we sample a solution which satisfies these
        constraints with Pareto-optimality? In particular, in a Pareto-optimal solution
        x*, no x*_j may be incremented and still satisfy the constraints.

        A GeneralUniformSelector object implements the approach described in
        K. Narayan. Towards Uniform Sampling of Pareto-Optimal Armies in StarCraft II.
        See the #gamebreaker channel for the paper.

        NOTE: once a GeneralUniformSelector object has been created, its properties cannot
        be changed once select() has been called.

        :param constraints: the list of constraints. Each element in the list is a tuple,
        (A_i, b_i) where A_i is a numpy array denoting the coefficients to the budget
        constraints and b_i is the limit to that budget constraint.

        :param seed: the random seed to use in sampling selectors according to the
        discrete probability distribution.
        """
        self.rng = _get_rng(seed)
        self._constraints = constraints
        self._subproblem_structures = []

        self._nvars = len(self._constraints[0][0])
        for constraint in self._constraints:
            if len(constraint) != 2:
                raise ValueError(
                    "Constraint must be of the form (a, b) where a is a list of "
                    "coefficients and b is a constant."
                )

            if len(constraint[0]) != self._nvars:
                raise ValueError(
                    f"Found constraints with varying lengths: {len(constraint[0])}, "
                    f"{len(constraint)}"
                )

            if type(constraint[1]) != int:
                raise ValueError(
                    f"Found a constraint (a, b), where b isn't an integer: {constraint}"
                )

            if constraint[1] < 0:
                raise ValueError("b must be non-negative for each constraint (a, b).")

    def _create_single_subproblem(self, constraints):
        # Create A, b
        As, bs = tuple(zip(*constraints))

        A = np.array(As)
        b = np.array(bs)

        # Attach slack variables to A
        A = np.hstack([A, np.zeros((len(constraints), len(constraints) - 1), dtype=np.int32),])
        A[1:, self._nvars :] = np.eye(len(constraints) - 1)

        # Deal with malformed values of A
        if np.min(A) < 0:
            raise ValueError("Encountered a negative cost in the problem specification.")

        # Detect whether we can create infinitely many units
        free_units = list(np.argwhere(A.sum(axis=0) == 0))
        if len(free_units) > 0:
            raise ValueError(
                "The following units can be produced infinitely given the "
                f"constraints: {free_units}"
            )

        return A, b

    def _tighten_constraint(self, tightish_constraint):
        # Sometimes, we need to make constraint a bit loose, particularly if it isn't
        # possible to attain equality. E.g., if we have the equation
        #
        # 5x + 7y + 7z + 7w = 15 (where = is tight-ish; meaning, we can't squeeze in
        #                         another unit)
        #
        # we want to expand the right hand side to include attainable values between
        # 11 and 15, inclusive; so, that's 14 and 15.
        a, b = tightish_constraint
        attainable_numbers = set([])
        for coeff in a:
            if coeff > 0:
                attainable_numbers.update(set(range(0, b + coeff, coeff)))

        # In the above example, we only want to keep values of the right hand side for
        # which it is impossible to squeeze in another unit. In the above case, the value
        # of the right hand side must be strictly greater than 15 - 5.
        tight_bs = [n for n in attainable_numbers if b - min([_a for _a in a if _a > 0]) < n <= b]
        assert len(tight_bs) > 0

        # Return a list of tight constraints
        return [(a.copy(), tight_b) for tight_b in tight_bs]

    def _reformat_constraint(self, constraint):
        # Determine the least number to multiply each side of the constraint by to turn
        # all coefficients into integers
        to_multiply = np.lcm.reduce(
            [Fraction(coeff).limit_denominator().denominator for coeff in constraint[0]]
            + [Fraction(constraint[1]).limit_denominator().denominator]
        )
        constraint = [
            [int(to_multiply * coeff) for coeff in constraint[0]],
            int(to_multiply * constraint[1]),
        ]

        # Reduce the constraint - not strictly necessary, but useful to generally keep
        # numbers low when we can help it
        to_divide = np.gcd.reduce(constraint[0] + [constraint[1]])
        return [
            [coeff // to_divide for coeff in constraint[0]],
            constraint[1] // to_divide,
        ]

    def _generate_subproblems(self):
        subproblems = []

        # Get rid of constraints with only 0s in the lhs and reformat those that pass.
        self._constraints = [
            self._reformat_constraint(c) for c in self._constraints if sum(c[0]) > 0
        ]

        # We aim to make constraint as tight as possible
        for tightish_constraint_ix, tightish_constraint in enumerate(self._constraints):
            # Sometimes, we need to make constraint a bit loose, particularly if it isn't
            # possible to attain equality. See _tighten_constraint for more details.
            all_other_constraints = (
                self._constraints[:tightish_constraint_ix]
                + self._constraints[tightish_constraint_ix + 1 :]
            )
            tight_constraints = self._tighten_constraint(tightish_constraint)

            cur_subproblems = [
                self._create_single_subproblem([tight_constraint] + all_other_constraints)
                for tight_constraint in tight_constraints
            ]
            subproblems.extend(cur_subproblems)

        # Eliminate subproblems where the right-hand side of the tight constraint is 0;
        # this can lead to generating solutions that are not Pareto-optimal. E.g., if we
        # have a mineral budget of 50 and a gas budget of 0 and are trying to generate
        # a Zerg army, in the case where we attempt to enforce tightness on the gas
        # budget and keep slackness on the mineral budget, we would end up with 2 valid
        # armies: (1) a single zergling, (2) two zerglings. Both solutions would force
        # tightness on the gas constraint and allow for slackness on the mineral
        # constraint. However, generating a single zergling is not Pareto optimal.
        return [subproblem for subproblem in subproblems if subproblem[1][0] > 0]

    def _generate_structure(self, A, b, col_ix=0, _P={}):
        # Only consider cases with positive b
        key = (col_ix,) + tuple(b)
        if np.min(b) < 0:
            return 0

        m, n = A.shape
        n_slacks = m - 1
        n_vars = n - n_slacks

        # When considering purely a subset of slack variables, we only ever have one
        # solution
        if col_ix >= n_vars:
            _P[key] = 1, []
            return 1

        # When considering a single actual variable and a subset of slack variables...
        # This strange expression is the right-most non-negative column index
        if col_ix == np.nonzero(A[0])[0][-1]:
            actual_var_value = int(b[0] / A[0, col_ix])
            checks = b - actual_var_value * A[:, col_ix]

            # (1) check whether the coefficient of the actual variable divides b[0]
            if b[0] % A[0, col_ix] != 0:
                _P[key] = 0, []
                return 0

            # (2) when substituted into the rest of the equations, check that the slacks
            # are all positive
            if np.min(checks) < 0:
                _P[key] = 0, []
                return 0

            # If both checks pass, then we have a single solution
            _P[key] = 1, []
            return 1

        # The recursive case
        cases = []
        cur_b = b.copy()
        t = 0

        while np.min(cur_b) >= 0:
            cur_b = np.array(b - t * A[:, col_ix], dtype=np.int32)

            key = (col_ix + 1,) + tuple(cur_b)
            if key in _P:
                count, _ = _P[key]
            else:
                count = self._generate_structure(A, cur_b, col_ix + 1, _P=_P)

            cases.append(count)
            t += 1

        count = sum(cases)
        _P[(col_ix,) + tuple(b)] = count, cases

        return count

    def _construct_structures(self):
        # Generate the various subproblems; each subproblem causes precisely one of the
        # constraints to become as tight as possible.
        subproblems = self._generate_subproblems()

        # Generate structures for each of the subproblems
        self._subproblem_structures = [{} for _ in subproblems]
        for ix, subproblem in enumerate(subproblems):
            total_count = self._generate_structure(
                subproblem[0], subproblem[1], _P=self._subproblem_structures[ix]
            )
            self._subproblem_structures[ix]["A"] = subproblem[0]
            self._subproblem_structures[ix]["b"] = subproblem[1]
            self._subproblem_structures[ix]["total_count"] = total_count

    def _sample(self, structure):
        A, b, _P = structure["A"], structure["b"], structure
        m, n = A.shape
        n_slacks = m - 1
        n_vars = n - n_slacks

        ret = np.zeros(n)
        b = b.copy()
        orig_b = b.copy()

        # Start with col_ix = 0 and the initial tuple, b
        for col_ix in range(n_vars - 1):
            state = (col_ix,) + tuple(b)

            # Draw a sample for the current column
            count, frequencies = _P[state]
            assert sum(frequencies) == count

            # By default, np.sum(pdf) should be 1; however, we divide by pdf.sum() for
            # cases where the sum is very close to 1 but not 1 due to numerical stability
            # issues.
            pdf = np.array(frequencies, dtype=np.float64) / np.array(count, dtype=np.float64)
            sample = self.rng.choice(list(range(len(frequencies))), p=pdf)

            ret[col_ix] = sample

            # Update b
            b -= A[:, col_ix] * sample

        # Fill in the final variable
        ret[n_vars - 1] = (orig_b[0] - A[0, : (n_vars - 1)].dot(ret[: (n_vars - 1)])) / A[
            0, n_vars - 1
        ]

        # Fill in the final variable and slacks
        for i in range(n_slacks):
            ret[n_vars + i] = (
                orig_b[1] - A[i + 1, : (n_vars + i + 1)].dot(ret[: (n_vars + i + 1)])
            ) / A[i + 1, n_vars + i]

        # Ensure that all elements in ret are integers
        for e in ret:
            assert np.floor(e) == np.ceil(e)

        return np.array(ret, dtype=np.int32)

    def select(self):
        # This is almost correct; there are very tiny oversampling issues when multiple
        # constraints are simultaneously tight. We need to use the principle of inclusion
        # exclusion to address this, but oh well. These oversampling issues typically end
        # up being very small.
        if not self._subproblem_structures:
            self._construct_structures()

        # Determine which substructure to sample from
        substructure_counts = [
            structure["total_count"] for structure in self._subproblem_structures
        ]

        # Return if there are no valid armies to sample
        if np.sum(substructure_counts) == 0:
            return

        num = np.array(substructure_counts)
        denom = np.sum(substructure_counts)

        substructure_ix = self.rng.choice(
            list(range(len(self._subproblem_structures))),
            p=np.array(num, dtype=np.float64) / np.array(denom, dtype=np.float64),
        )

        # Compute a sample
        return self._sample(self._subproblem_structures[substructure_ix])


# from fractions import Fraction
#
# f = Fraction(3.1416)
# print(f)
# x = f.limit_denominator(30)
# print(x)
#

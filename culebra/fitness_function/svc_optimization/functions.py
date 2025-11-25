# This file is part of culebra.
#
# Culebra is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Culebra is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# Culebra. If not, see <http://www.gnu.org/licenses/>.
#
# This work is supported by projects PGC2018-098813-B-C31 and
# PID2022-137461NB-C31, both funded by the Spanish "Ministerio de Ciencia,
# Innovación y Universidades" and by the European Regional Development Fund
# (ERDF).

"""Fitness functions for parameter optimization problems."""

from __future__ import annotations

from typing import Optional, Tuple
from collections.abc import Sequence

from culebra.abc import Fitness
from culebra.fitness_function.dataset_score import (
    KappaIndex as DatasetKappaIndex,
    Accuracy as DatasetAccuracy
)
from culebra.fitness_function.svc_optimization.abc import (
    SVCScorer,
    RBFSVCScorer
)
from culebra.solution.parameter_optimization import Solution


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class C(SVCScorer):
    """Minimization of the C hyperparameter of RBF SVCs."""

    @property
    def obj_weights(self) -> Tuple[int, ...]:
        """Objective weights.

        Minimize C.

        :return: (-1, )
        :rtype: tuple[int]
        """
        return (-1, )

    @property
    def obj_names(self) -> Tuple[str, ...]:
        """Objective names.

        :return: ("C",)
        :rtype: tuple[str]
        """
        return ("C",)

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.solution.parameter_optimization.Solution
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: int
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives:
            ~collections.abc.Sequence[~culebra.abc.Solution]
        :return: The fitness for *sol*
        :rtype: ~culebra.abc.Fitness
        :raises ValueError: If *sol* is not evaluable
        """
        if not self.is_evaluable(sol):
            raise ValueError("The solution is not evaluable")

        # Set the value of C
        sol.fitness.update_value(sol.values.C, self.index)

        return sol.fitness


class KappaIndex(RBFSVCScorer, DatasetKappaIndex):
    """Single-objective fitness function for classification problems.

    Calculate the Kohen's Kappa index.
    """


class Accuracy(RBFSVCScorer, DatasetAccuracy):
    """Single-objective fitness function for classification problems.

    Calculate the accuracy.
    """


# Exported symbols for this module
__all__ = [
    'C',
    'KappaIndex',
    'Accuracy'
]

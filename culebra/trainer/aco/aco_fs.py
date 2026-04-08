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

"""Implementation of ACO algorithms for feature selection problems."""

from __future__ import annotations

from collections.abc import Sequence
import math
from itertools import combinations

import numpy as np

from culebra.abc import Base
from culebra.solution.feature_selection import Ant
from culebra.trainer.aco import DEFAULT_PHEROMONE_DEPOSIT_WEIGHT
from culebra.trainer.aco.abc import (
    ReseteablePheromoneBasedACO,
    ACOTSP,
    ACOFS
)



__author__ = 'Jesús González & Alberto Ortega'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es & aoruiz@ugr.es'
__status__ = 'Development'


class ACOFSConvergenceDetector(Base):
    """Detect the convergence of an :attr:`~culebra.trainer.aco.abc.ACOFS`
    instance.
    """

    def __init__(
        self, convergence_check_freq: int | None = None
    ) -> None:
        """Create a convergence detector.

        :param convergence_check_freq: Convergence assessment frequency. If
            omitted,
            :attr:`~culebra.trainer.aco.DEFAULT_CONVERGENCE_CHECK_FREQ`
            will be used. Defaults to :data:`None`
        :type convergence_check_freq: int
        """
        self.convergence_check_freq = convergence_check_freq
        self.last_pheromone = None

    convergence_check_freq = (
        ReseteablePheromoneBasedACO.convergence_check_freq
    )

    def has_converged(self, trainer) -> bool:
        """Detect if the trainer has converged.

        :param trainer: The trainer
        :type trainer: ~culebra.trainer.aco.abc.ACOFS

        :return: :data:`True` if the trainer has converged
        :rtype: bool
        """
        convergence = False
        if trainer.current_iter % self.convergence_check_freq == 0:
            if trainer.current_iter != 0:
                diff = np.sum(
                    np.abs(
                        trainer.pheromone[0] - self.last_pheromone[0]
                    )
                )
                if np.isclose(diff, 0):
                    convergence = True

            self.last_pheromone = trainer.pheromone

        return convergence


class ACOFS1D(ACOFS):
    """Abstract base class for all the vector pheromone ACO-FS trainers."""

    @property
    def pheromone_shapes(self) -> tuple[tuple[int, int], ...]:
        """Shape of the pheromone matrices.

        :rtype: tuple[tuple[int]]
        """
        return (
            ((self.species.num_feats, ),) * self.num_pheromone_matrices
        )

    @property
    def heuristic_shapes(self) -> tuple[tuple[int, int], ...]:
        """Shape of the heuristic matrices.

        :rtype: tuple[tuple[int]]
        """
        return (
            ((self.species.num_feats, ),) * self.num_heuristic_matrices
        )

    @property
    def _default_heuristic(self) -> tuple[np.ndarray[float], ...]:
        """Default heuristic matrices.

        :return: An all-ones vector for each heuristic matrix
        :rtype: tuple[~numpy.ndarray[float]]
        """
        return tuple(
            np.ones(shape, dtype=np.float64)
            for shape in self.heuristic_shapes
        )

    def _ant_choice_info(self, ant: Ant) -> np.ndarray[float]:
        """Return the choice info to obtain the next node the ant will visit.

        All the nodes banned by the
        :attr:`~culebra.trainer.aco.ACOFS1D.species`, along with all
        the previously visited nodes are discarded.

        :param ant: The ant
        :type ant: ~culebra.solution.feature_selection.Ant
        :rtype: ~numpy.ndarray[float]
        """
        ant_choice_info = np.copy(self.choice_info)

        # Discard the unfeasible nodes
        ant_choice_info[self._unfeasible_nodes(ant)] = 0

        return ant_choice_info

    def _deposit_pheromone(
        self,
        ants: Sequence[Ant],
        weight: float = DEFAULT_PHEROMONE_DEPOSIT_WEIGHT
    ) -> None:
        """Make some ants deposit weighted pheromone.

        The pheromone amount deposited by each ant is equally divided across
        all possible feature pair combinations derived from its set of
        selected features.

        :param ants: The ants
        :type ants:
            ~collections.abc.Sequence[~culebra.solution.feature_selection.Ant]
        :param weight: Weight for the pheromone. Defaults to
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_DEPOSIT_WEIGHT`
        :type weight: float
        """
        for ant in ants:
            for pher, pher_amount in zip(
                self.pheromone, self._pheromone_amount(ant)
            ):
                for feat in ant.path:
                    pher[feat] += pher_amount * weight


class ACOFS2D(ACOFS):
    """Abstract base class for all the 2D pheromone matrix ACO-FS trainers."""

    @property
    def pheromone_shapes(self) -> tuple[tuple[int, int], ...]:
        """Shape of the pheromone matrices.

        :rtype: tuple[tuple[int]]
        """
        return (
            ((self.species.num_feats, ) * 2,) * self.num_pheromone_matrices
        )

    @property
    def heuristic_shapes(self) -> tuple[tuple[int, int], ...]:
        """Shape of the heuristic matrices.

        :rtype: tuple[tuple[int]]
        """
        return (
            ((self.species.num_feats, ) * 2,) * self.num_heuristic_matrices
        )

    @property
    def _default_heuristic(self) -> tuple[np.ndarray[float], ...]:
        """Default heuristic matrices.

        :return: An all-ones matrix with a zero diagonal for each heuristic
            matrix
        :rtype: tuple[~numpy.ndarray[float]]
        """
        def create_hollow_matrix(shape: tuple[int, int]) -> np.ndarray:
            """Create an all-ones matrix with a zero diagonal.

            :param shape: Matrix shape
            :type shape: tuple[int, int]
            :return: The matrix
            :rtype: ~numpy.ndarray
            """
            arr = np.ones(shape, dtype=np.float64)
            np.fill_diagonal(arr, 0)
            return arr

        return tuple(
            create_hollow_matrix(shape)
            for shape in self.heuristic_shapes
        )

    # Use the same implementation as ACOTSP for _ant_choice_info
    _ant_choice_info = ACOTSP._ant_choice_info

    def _deposit_pheromone(
        self,
        ants: Sequence[Ant],
        weight: float = DEFAULT_PHEROMONE_DEPOSIT_WEIGHT
    ) -> None:
        """Make some ants deposit weighted pheromone.

        The pheromone amount deposited by each ant is equally divided across
        all possible feature pair combinations derived from its set of
        selected features.

        A symmetric problem is assumed. Thus if (*i*, *j*) is an arc in an
        ant's path, arc (*j*, *i*) is also incremented the by same amount.

        :param ants: The ants
        :type ants:
            ~collections.abc.Sequence[~culebra.solution.feature_selection.Ant]
        :param weight: Weight for the pheromone. Defaults to
            :attr:`~culebra.trainer.aco.DEFAULT_PHEROMONE_DEPOSIT_WEIGHT`
        :type weight: float
        """
        for ant in ants:
            if len(ant.path) > 1:
                # All the combinations of two features from those in the path
                indices = combinations(ant.path, 2)
                num_combinations = math.comb(len(ant.path), 2)
                delta = weight / num_combinations

                # Deposit the pheromone
                for pher, pher_amount in zip(
                    self.pheromone, self._pheromone_amount(ant)
                ):
                    amount_per_combination = pher_amount * delta
                    for (i, j) in indices:
                        pher[i][j] += amount_per_combination
                        pher[j][i] += amount_per_combination


__all__ = [
    'ACOFSConvergenceDetector',
    'ACOFS1D',
    'ACOFS2D'
]

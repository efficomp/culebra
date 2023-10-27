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
# This work was supported by project PGC2018-098813-B-C31 (Spanish "Ministerio
# de Ciencia, Innovación y Universidades"), and by the European Regional
# Development Fund (ERDF).

"""Fitness functions for parameter optimization problems.

This sub-module provides several fitness functions intended to optimize the
Support Vector Classifier (SVC) hyperparameters for a given dataset. The
following fitness functions are provided:

  * :py:class:`~culebra.fitness_function.svc_optimization.C`: Dummy
    single-objective function that minimizes the regularization parameter *C*
    of a SVM-based classifier with RBF kernels.

  * :py:class:`~culebra.fitness_function.svc_optimization.KappaIndex`:
    Single-objective function that maximizes the Kohen's Kappa index for a
    SVM-based classifier with RBF kernels.

  * :py:class:`~culebra.fitness_function.svc_optimization.KappaC`: Bi-objective
    function that tries to both maximize the Kohen's Kappa index and minimize
    the regularization parameter *C* of a SVM-based classifier with RBF
    kernels.
"""

from __future__ import annotations

from typing import Optional, Tuple
from collections.abc import Sequence

from sklearn.metrics import cohen_kappa_score

from culebra.abc import Fitness
from culebra.fitness_function import DEFAULT_THRESHOLD
from culebra.fitness_function.abc import RBFSVCFitnessFunction
from culebra.solution.parameter_optimization import Solution


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2023, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.3.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class C(RBFSVCFitnessFunction):
    """Minimization of the C hyperparameter of RBF SVCs."""

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.solution.svc_optimization.C.evaluate` method within
        a :py:class:`~culebra.solution.parameter_optimization.Solution`.
        """

        weights = (-1.0,)
        """Minimize C."""

        names = ("C",)
        """Name of the objective."""

        thresholds = (DEFAULT_THRESHOLD,)
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol:
            :py:class:`~culebra.solution.parameter_optimization.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Return the value of C
        return (sol.values.C,)


class KappaIndex(RBFSVCFitnessFunction):
    """Maximize the Kohen's Kappa index."""

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.solution.svc_optimization.C.evaluate` method within
        a :py:class:`~culebra.solution.parameter_optimization.Solution`.
        """

        weights = (1.0,)
        """Maximizes the valiation Kappa index."""

        names = ("Kappa",)
        """Name of the objective."""

        thresholds = (DEFAULT_THRESHOLD,)
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol:
            :py:class:`~culebra.solution.parameter_optimization.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        # Set the classifier hyperparameters
        hyperparams = sol.values
        self.classifier.C = hyperparams.C
        self.classifier.gamma = hyperparams.gamma

        # Get the training and test data
        training_data, test_data = self._final_training_test_data()

        # Train and get the outputs for the validation data
        outputs_pred = self.classifier.fit(
            training_data.inputs, training_data.outputs
        ).predict(test_data.inputs)
        kappa = cohen_kappa_score(test_data.outputs, outputs_pred)

        return (kappa,)


class KappaC(KappaIndex, C):
    """Bi-objective fitness class for feature selection.

    Maximizes the Kohen's Kappa index and minimizes the C regularization
    htyperparameter.
    """

    class Fitness(Fitness):
        """Fitness class.

        Handles the values returned by the
        :py:meth:`~culebra.solution.svc_optimization.C.evaluate` method within
        a :py:class:`~culebra.solution.parameter_optimization.Solution`.
        """

        weights = KappaIndex.Fitness.weights + C.Fitness.weights
        """Maximizes the Kohen's Kappa index and minimizes the C regularization
        hyperparameter.
        """

        names = KappaIndex.Fitness.names + C.Fitness.names
        """Name of the objectives."""

        thresholds = (
            KappaIndex.Fitness.thresholds + C.Fitness.thresholds
        )
        """Similarity threshold for fitness comparisons."""

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Tuple[float, ...]:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol:
            :py:class:`~culebra.solution.parameter_optimization.Solution`
        :param index: Index where *sol* should be inserted in the
            representatives sequence to form a complete solution for the
            problem. Only used by cooperative problems
        :type index: :py:class:`int`, ignored
        :param representatives: Representative solutions of each species
            being optimized. Only used by cooperative problems
        :type representatives: :py:class:`~collections.abc.Sequence` of
            :py:class:`~culebra.abc.Solution`, ignored
        :return: The fitness of *sol*
        :rtype: :py:class:`tuple` of :py:class:`float`
        """
        return KappaIndex.evaluate(self, sol) + C.evaluate(self, sol)


# Exported symbols for this module
__all__ = [
    'RBFSVCFitnessFunction',
    'C',
    'KappaIndex',
    'KappaC'
]

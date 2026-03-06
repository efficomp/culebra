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

"""Fitness functions for parameter optimization problems.

This sub-module provides several abstract fitness functions intended to
optimize the Support Vector Classifier (SVC) hyperparameters for a given
dataset:

* :class:`~culebra.fitness_func.svc_optimization.abc.RBFSVCScorer`:
  Is centered on the hyperparameters optimization of SVM-based classifiers
  with RBF kernels.
* :class:`~culebra.fitness_func.svc_optimization.abc.SVCScorer`:
  Abstract fitness function for the hyperparameters optimization of SVM-based
  classifiers.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

from culebra.checker import check_instance
from culebra.abc import Fitness
from culebra.fitness_func.abc import SingleObjectiveFitnessFunction
from culebra.fitness_func.dataset_score.abc import ClassificationScorer
from culebra.solution.parameter_optimization import Solution


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class SVCScorer(SingleObjectiveFitnessFunction):
    """Single objective function for SVC optimization problems."""

    def is_evaluable(self, sol: Solution) -> bool:
        """Assess the evaluability of a solution.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.solution.parameter_optimization.Solution
        :return: :data:`True` if the solution can be evaluated
        :rtype: bool
        :raises NotImplementedError: If has not been overridden
        """
        if not isinstance(sol, Solution):
            return False

        param_names = sol.values._fields
        if 'C' in param_names and 'gamma' in param_names:
            return True

        return False


class RBFSVCScorer(ClassificationScorer, SVCScorer):
    """Abstract base class fitness function for RBF SVC optimization."""

    @property
    def _default_classifier(self) -> SVC:
        """Default classifier.

        :return: An SVC with RBF kernels
        :rtype: ~sklearn.svm.SVC
        """
        return SVC(kernel='rbf')

    @ClassificationScorer.classifier.setter
    def classifier(self, value: ClassifierMixin | None) -> None:
        """Set a classifier.

        The classifier must be an instance of :class:`~sklearn.svm.SVC` with
        RBF kernels.

        :param value: The classifier. If set to :data:`None`, an instance of
            :class:`~sklearn.svm.SVC` with RBF kernels is used
        :type value: ~sklearn.svm.SVC
        :raises TypeError: If set to a value which is not a valid
            :class:`~sklearn.svm.SVC` instance
        :raises ValueError: If the classifier has not RBF kernels
        """
        if value is not None:
            check_instance(value, "classifier", cls=SVC)

            if value.kernel != 'rbf':
                raise ValueError(
                    f"The classifier has not RBF kernels: {value}"
                )

        ClassificationScorer.classifier.fset(self, value)

    def evaluate(
        self,
        sol: Solution,
        index: int | None = None,
        cooperators: Sequence[Solution | None] | None = None
    ) -> Fitness:
        """Evaluate a solution.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.solution.parameter_optimization.Solution
        :param index: Index where *sol* should be inserted in the cooperators
            sequence to form a complete solution for the problem
        :type index: int
        :param cooperators: Cooperators of each species being optimized
        :type cooperators:
            ~collections.abc.Sequence[~culebra.abc.Solution]
        :return: The fitness for *sol*
        :rtype: ~culebra.abc.Fitness
        :raises ValueError: If *sol* is not evaluable
        """
        if not self.is_evaluable(sol):
            raise ValueError("The solution is not evaluable")

        # Set the classifier hyperparameters
        hyperparams = sol.values
        self.classifier.C = hyperparams.C
        self.classifier.gamma = hyperparams.gamma

        return ClassificationScorer.evaluate(self, sol, index, cooperators)


# Exported symbols for this module
__all__ = [
    'SVCScorer',
    'RBFSVCScorer'
]

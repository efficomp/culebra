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

  * :py:class:`~culebra.fitness_function.svc_optimization.abc.SVCScorer`:
    Abstract fitness function for thehyperparameters optimization of SVM-based
    classifiers.

  * :py:class:`~culebra.fitness_function.svc_optimization.abc.RBFSVCScorer`:
    Is centered on the hyperparameters optimization of SVM-based classifiers
    with RBF kernels.
"""

from __future__ import annotations

from typing import Optional
from collections.abc import Sequence

from sklearn.base import ClassifierMixin
from sklearn.svm import SVC

from culebra.checker import check_instance
from culebra.abc import Fitness
from culebra.fitness_function.abc import SingleObjectiveFitnessFunction
from culebra.fitness_function.dataset_score.abc import ClassificationScorer
from culebra.solution.parameter_optimization import Solution
from culebra.tools import Dataset


__author__ = 'Jesús González'
__copyright__ = 'Copyright 2025, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.4.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class SVCScorer(SingleObjectiveFitnessFunction):
    """Single objective function for SVC optimization problems."""

    def is_evaluable(self, sol: Solution) -> bool:
        """Return :py:data:`True` if the solution can be evaluated.

        :param sol: Solution to be evaluated.
        :type sol:
            :py:class:`~culebra.solution.parameter_optimization.Solution`
        """
        if not isinstance(sol, Solution):
            return False

        param_names = sol.values._fields
        if 'C' in param_names and 'gamma' in param_names:
            return True

        return False


class RBFSVCScorer(ClassificationScorer, SVCScorer):
    """Abstract base class fitness function for RBF SVC optimization."""

    def __init__(
        self,
        training_data: Dataset,
        test_data: Optional[Dataset] = None,
        cv_folds: Optional[int] = None,
        classifier: Optional[ClassifierMixin] = None,
        index: Optional[int] = None
    ) -> None:
        """Construct the fitness function.

        If *test_data* are provided, the whole *training_data* are used to
        train. Otherwise, a *k*-fold cross-validation is applied.

        :param training_data: The training dataset
        :type training_data: :py:class:`~culebra.tools.Dataset`
        :param test_data: The test dataset, defaults to :py:data:`None`
        :type test_data: :py:class:`~culebra.tools.Dataset`, optional
        :param cv_folds: The number of folds for *k*-fold cross-validation.
            If omitted,
            :py:attr:`~culebra.fitness_function.dataset_score.DEFAULT_CV_FOLDS`
            is used. Defaults to :py:data:`None`
        :type cv_folds: :py:class:`int`, optional
        :param classifier: The classifier. If set to :py:data:`None`,
            a :py:class:`~sklearn.svm.SVC` with RBF kernels will be used.
            Defaults to :py:data:`None`
        :type classifier: :py:class:`~sklearn.svm.SVC` with
            RBF kernels, optional
        :param index: Index of this objective when it is used for
            multi-objective fitness functions
        :type index: :py:class:`int`, optional

        :raises RuntimeError: If the number of objectives is not 1
        :raises TypeError: If *training_data* or *test_data* is an invalid
            dataset
        :raises TypeError: If *cv_folds* is not an integer value
        :raises ValueError: If *cv_folds* is not positive
        :raises TypeError: If *classifier* is not a valid
            :py:class:`~sklearn.svm.SVC` instance
        :raises TypeError: If *index* is not an integer number
        :raises ValueError: If *index* is not positive
        """
        # Init the superclass
        super().__init__(
            training_data,
            test_data,
            cv_folds,
            classifier,
            index
        )

    @property
    def classifier(self) -> SVC:
        """Get and set the classifier applied within this fitness function.

        :getter: Return the classifier
        :setter: Set a new classifier
        :type: Any subclass of :py:class:`~sklearn.svm.SVC` with RBF kernels
        :raises TypeError: If set to a value which is not a valid
            :py:class:`~sklearn.svm.SVC` instance
        :raises ValueError: If the classifier has not RBF kernels
        """
        return self._classifier

    @classifier.setter
    def classifier(self, value: ClassifierMixin | None) -> None:
        """Set a classifier.

        :param value: The classifier. If set to :py:data:`None`,
            :py:class:`~sklearn.svm.SVC with RBF kernels is chosen
        :type: Any subclass of :py:class:`~sklearn.svm.SVC` with RBF kernels
        :raises TypeError: If set to a value which is not a valid
            :py:class:`~sklearn.svm.SVC` instance
        :raises ValueError: If the classifier has not RBF kernels
        """
        if value is not None:
            value = check_instance(
                value, "classifier", cls=SVC
            )

            if value.kernel != 'rbf':
                raise ValueError(
                    f"The classifier has not RBF kernels: {value}"
                )
        else:
            value = SVC(kernel='rbf')

        self._classifier = value

    def evaluate(
        self,
        sol: Solution,
        index: Optional[int] = None,
        representatives: Optional[Sequence[Solution]] = None
    ) -> Fitness:
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
        :return: The fitness for *sol*
        :rtype: :py:class:`~culebra.abc.Fitness`
        :raises ValueError: If *sol* is not evaluable
        """
        if not self.is_evaluable(sol):
            raise ValueError("The solution is not evaluable")

        # Set the classifier hyperparameters
        hyperparams = sol.values
        self.classifier.C = hyperparams.C
        self.classifier.gamma = hyperparams.gamma

        return ClassificationScorer.evaluate(self, sol, index, representatives)


# Exported symbols for this module
__all__ = [
    'SVCScorer',
    'RBFSVCScorer'
]

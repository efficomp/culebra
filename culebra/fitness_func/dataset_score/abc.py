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

"""Fitness functions related to dataset scoring.

This sub-module provides several abstract fitness functions to score
dataset-related problems:

* :class:`~culebra.fitness_func.dataset_score.abc.ClassificationScorer`:
  Lets defining dataset classification-related fitness functions
* :class:`~culebra.fitness_func.dataset_score.abc.DatasetScorer`:
  Allows the definition of dataset-related fitness functions.
"""

from __future__ import annotations

from abc import abstractmethod

from collections.abc import Sequence
from copy import deepcopy

import secrets
from sklearn.base import ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer

from culebra.abc import Solution
from culebra.checker import check_int, check_instance, check_bool
from culebra.fitness_func.abc import SingleObjectiveFitnessFunction
from culebra.fitness_func.dataset_score import (
    DEFAULT_CV_NUM_FOLDS,
    DEFAULT_CV_FIXED_FOLDS
)
from culebra.tools import Dataset

__author__ = 'Jesús González'
__copyright__ = 'Copyright 2026, EFFICOMP'
__license__ = 'GNU GPL-3.0-or-later'
__version__ = '0.6.1'
__maintainer__ = 'Jesús González'
__email__ = 'jesusgonzalez@ugr.es'
__status__ = 'Development'


class DatasetScorer(SingleObjectiveFitnessFunction):
    """Abstract base fitness function for dataset evaluation problems."""

    def __init__(
        self,
        training_data: Dataset,
        test_data: Dataset | None = None,
        cv_num_folds: int | None = None,
        cv_fixed_folds: bool | None = None,
        index: int | None = None
    ) -> None:
        """Construct the fitness function.

        If *test_data* are provided, the whole *training_data* are used to
        train. Otherwise, a *k*-fold cross-validation is applied.

        :param training_data: The training dataset
        :type training_data: ~culebra.tools.Dataset
        :param test_data: The test dataset, defaults to :data:`None`
        :type test_data: ~culebra.tools.Dataset
        :param cv_num_folds: The number of folds for *k*-fold cross-validation.
            If omitted,
            :attr:`~culebra.fitness_func.dataset_score.abc.DatasetScorer._default_cv_num_folds`
            is used. Defaults to :data:`None`
        :type cv_num_folds: int
        :param cv_fixed_folds: If :data:`True`, the same folds are used across
            all evaluations, ensuring deterministic results. If :data:`False`,
            new folds are generated randomly for each evaluation, introducing
            variability in the results. If omitted,
            :attr:`~culebra.fitness_func.dataset_score.abc.DatasetScorer._default_cv_fixed_folds`
            is used. Defaults to :data:`None`
        :type cv_fixed_folds: bool
        :param index: Index of this objective when it is used for
            multi-objective fitness functions
        :type index: int
        :raises RuntimeError: If the number of objectives is not 1
        :raises TypeError: If *training_data* or *test_data* is an invalid
            dataset
        :raises TypeError: If *cv_num_folds* is not an integer value
        :raises ValueError: If *cv_num_folds* is not positive
        :raises TypeError: If *cv_fixed_folds* is not boolean
        :raises TypeError: If *index* is not an integer number
        :raises ValueError: If *index* is not positive
        """
        # Init the superclass
        super().__init__(index)

        # The cv_splitter can't be created until
        # cv_num_folds and cv_fixed_folds are defined
        # _init_complete become True when both are ready
        self._init_complete = False

        # Set the attributes to default values
        self.training_data = training_data
        self.test_data = test_data
        self.cv_num_folds = cv_num_folds
        self.cv_fixed_folds = cv_fixed_folds

        self._init_complete = True
        self._reset_cv_splitter()

    @property
    def training_data(self) -> Dataset:
        """Training dataset.

        :rtype: ~culebra.tools.Dataset
        :setter: Set a new training dataset
        :param value: The new training dataset
        :type value: ~culebra.tools.Dataset
        :raises TypeError: If set to an invalid dataset
        """
        return self._training_data

    @training_data.setter
    def training_data(self, value: Dataset) -> None:
        """Set a new training dataset.

        :param value: The new training dataset
        :type value: ~culebra.tools.Dataset
        :raises TypeError: If set to an invalid dataset
        """
        self._training_data = check_instance(
            value, "training data", cls=Dataset
        )

    @property
    def test_data(self) -> Dataset | None:
        """Test dataset.

        If set to :data:`None`, a *k*-fold cross-validation is applied.

        :rtype: ~culebra.tools.Dataset

        :setter: Set a new test dataset
        :param value: The new test dataset
        :type value: ~culebra.tools.Dataset
        :raises TypeError: If set to an invalid dataset
        """
        return self._test_data

    @test_data.setter
    def test_data(self, value: Dataset | None) -> None:
        """Set a new test dataset.

        :param value: The new test dataset
        :type value: ~culebra.tools.Dataset
        :raises TypeError: If set to an invalid dataset
        """
        self._test_data = None if value is None else check_instance(
            value, "test data", cls=Dataset
        )

    @property
    def _default_cv_num_folds(self) -> int:
        """Default number of folds for cross-validation.

        :return:
            :attr:`~culebra.fitness_func.dataset_score.DEFAULT_CV_NUM_FOLDS`
        :rtype: int
        """
        return DEFAULT_CV_NUM_FOLDS

    @property
    def cv_num_folds(self) -> int:
        """Number of cross-validation folds.

        :rtype: int

        :setter: Set a new value for the number of cross-validation folds
        :param value: A positive integer value. If set to :data:`None`,
            :attr:`~culebra.fitness_func.dataset_score.abc.DatasetScorer._default_cv_num_folds`
            is assumed
        :type value: int
        :raises TypeError: If *value* is not an integer value
        :raises ValueError: If *value* is not positive
        """
        return (
            self._default_cv_num_folds
            if self._cv_num_folds is None
            else self._cv_num_folds
        )

    @cv_num_folds.setter
    def cv_num_folds(self, value: int | None) -> None:
        """Set a value for the number of cross-validation folds.

        :param value: A positive integer value. If set to :data:`None`,
            :attr:`~culebra.fitness_func.dataset_score.abc.DatasetScorer._default_cv_num_folds`
            is assumed
        :type value: int
        :raises TypeError: If *value* is not an integer value
        :raises ValueError: If *value* is not positive
        """
        self._cv_num_folds = (
            None if value is None else check_int(
                value, "number of cross-validation folds", gt=0
            )
        )
        self._reset_cv_splitter()

    @property
    def _default_cv_fixed_folds(self) -> bool:
        """Default value for the fixed folds flag.

        :return:
            :attr:`~culebra.fitness_func.dataset_score.DEFAULT_CV_FIXED_FOLDS`
        :rtype: bool
        """
        return DEFAULT_CV_FIXED_FOLDS

    @property
    def cv_fixed_folds(self) -> bool:
        """Whether the folds are fixed or not across evaluations.

        :rtype: bool

        :setter: Set a new value for the fixed folds flag. If set to
            :data:`None`,
            :attr:`~culebra.fitness_func.dataset_score.abc.DatasetScorer._default_cv_fixed_folds`
            is assumed
        :type value: bool
        :raises TypeError: If *value* is not boolean
        """
        return (
            self._default_cv_fixed_folds
            if self._cv_fixed_folds is None
            else self._cv_fixed_folds
        )

    @cv_fixed_folds.setter
    def cv_fixed_folds(self, value: bool | None) -> None:
        """Set the fixed folds flag.

        :param value: :data:`True` to keep the folds fixed across evaluations.
            If set to :data:`None`,
            :attr:`~culebra.fitness_func.dataset_score.abc.DatasetScorer._default_cv_fixed_folds`
            is assumed
        :type value: bool
        :raises TypeError: If *value* is not boolean
        """
        self._cv_fixed_folds = (
            None if value is None else check_bool(value, "cv_fixed_folds")
        )
        self._reset_cv_splitter()

    @property
    def cv_splitter(self) -> StratifiedKFold:
        """Return the dataset splitter.
        
        :rtype: ~sklearn.model_selection.StratifiedKFold
        """
        return self._cv_splitter

    def _reset_cv_splitter(self) -> None:
        """Reset the dataset splitter."""
        # The cv_splitter can't be created until
        # cv_num_folds and cv_fixed_folds are defined
        # _init_complete become True when both are ready
        if not self._init_complete:
            return

        if self.cv_fixed_folds is True:
            random_seed = secrets.randbits(32)
        else:
            random_seed = None

        self._cv_splitter = StratifiedKFold(
            n_splits=self.cv_num_folds,
            shuffle=True,
            random_state=random_seed
        )

    @property
    @abstractmethod
    def _worst_score(self) -> float:
        """Worst achievable score.

        This property must be overridden by subclasses to return a correct
        value.

        :rtype: float
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The path property has not been implemented in the "
            f"{self.__class__.__name__} class"
        )

    @staticmethod
    @abstractmethod
    def _score(
        outputs: Sequence[float],
        outputs_pred: Sequence[float],
        **kwargs: dict
    ) -> float:
        """Score function to be used in the evaluation.

        This method must be overridden by subclasses to return a correct
        value.
        """
        raise NotImplementedError(
            "The scorer method has not been implemented")

    def _final_training_test_data(
        self,
        sol: Solution
    ) -> tuple[Dataset, Dataset]:
        """Get the final training and test data.

        :param sol: Solution to be evaluated. It may influence the final
            datasets
        :type sol: ~culebra.abc.Solution

        :return: The final training and test datasets
        :rtype: tuple[~culebra.tools.Dataset]
        """
        return self.training_data, self.test_data

    @abstractmethod
    def _evaluate_train_test(
        self,
        sol: Solution,
        training_data: Dataset,
        test_data: Dataset
    ) -> tuple[float, ...]:
        """Evaluate a solution.

        Neither the solution nor its fitness should be modified.

        This method must be overridden by subclasses to return a correct
        value.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :param training_data: The training dataset
        :type training_data: ~culebra.tools.Dataset
        :param test_data: The test dataset
        :type test_data: ~culebra.tools.Dataset
        :return: The fitness values for *sol*
        :rtype: tuple[float, ...]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _evaluate_train_test method has not been implemented in the "
            f"{self.__class__.__name__} class")

    @abstractmethod
    def _evaluate_kfcv(
        self,
        sol: Solution,
        training_data: Dataset,
    ) -> tuple[float, ...]:
        """Evaluate a solution.

        A *k*-fold cross-validation is applied using the *training_data* with
        :attr:`~culebra.fitness_func.dataset_score.abc.DatasetScorer.cv_num_folds`
        folds.

        Neither the solution nor its fitness should be modified.

        This method must be overridden by subclasses to return a correct
        value.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :param training_data: The training dataset
        :type training_data: ~culebra.tools.Dataset
        :return: The fitness values for *sol*
        :rtype: tuple[float, ...]
        :raises NotImplementedError: If has not been overridden
        """
        raise NotImplementedError(
            "The _evaluate_kfcv method has not been implemented in the "
            f"{self.__class__.__name__} class")

    def evaluate(
        self,
        sol: Solution,
        index: int | None = None,
        cooperators: Sequence[Solution | None] | None = None
    ) -> tuple[float, ...]:
        """Evaluate a solution.

        Neither the solution nor its fitness should be modified.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :param index: Index where *sol* should be inserted in the cooperators
            sequence to form a complete solution for the problem. Only used by
            cooperative problems
        :type index: int
        :param cooperators: Cooperators of each species being optimized. Only
            used by cooperative problems
        :type cooperators:
            ~collections.abc.Sequence[~culebra.abc.Solution]
        :return: The fitness values for *sol*
        :rtype: tuple[float, ...]
        :raises ValueError: If *sol* is not evaluable
        """
        if not self.is_evaluable(sol):
            raise ValueError("The solution is not evaluable")

        # Obtain the final training and test data
        training_data, test_data = self._final_training_test_data(sol)

        # If test data are available, train with the whole training data
        # and test with the test data
        if test_data is not None:
            return self._evaluate_train_test(sol, training_data, test_data)

        return self._evaluate_kfcv(sol, training_data)

    def __copy__(self) -> DatasetScorer:
        """Shallow copy the fitness function.

        :return: The copied fitness function
        :rtype: ~culebra.fitness_func.dataset_score.abc.DatasetScorer
        """
        cls = self.__class__
        result = cls(self.training_data)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo: dict) -> DatasetScorer:
        """Deepcopy the fitness function.

        :param memo: Fitness function attributes
        :type memo: dict
        :return: The copied fitness function
        :rtype: ~culebra.fitness_func.dataset_score.abc.DatasetScorer
        """
        cls = self.__class__
        result = cls(
            deepcopy(self.training_data, memo)
        )
        result.__dict__.update(
            deepcopy(
                self.__dict__,
                memo | {id(self.training_data): result.training_data}
            )
        )
        return result

    def __reduce__(self) -> tuple:
        """Reduce the fitness function.

        :return: The reduction
        :rtype: tuple
        """
        return (self.__class__, (self.training_data,), self.__dict__)

    @classmethod
    def __fromstate__(cls, state: dict) -> DatasetScorer:
        """Return a dataset scorer from a state.

        :param state: The state
        :type state: ~dict
        :return: The fitness function
        :rtype: ~culebra.fitness_func.dataset_score.abc.DatasetScorer
        """
        obj = cls(state['_training_data'])
        obj.__setstate__(state)
        return deepcopy(obj)


class ClassificationScorer(DatasetScorer):
    """Abstract base fitness function for dataset classification problems."""

    def __init__(
        self,
        training_data: Dataset,
        test_data: Dataset | None = None,
        cv_num_folds: int | None = None,
        cv_fixed_folds: bool | None = None,
        classifier: ClassifierMixin | None = None,
        index: int | None = None
    ) -> None:
        """Construct the fitness function.

        If *test_data* are provided, the whole *training_data* are used to
        train. Otherwise, a *k*-fold cross-validation is applied.

        :param training_data: The training dataset
        :type training_data: ~culebra.tools.Dataset
        :param test_data: The test dataset, defaults to :data:`None`
        :type test_data: ~culebra.tools.Dataset
        :param cv_num_folds: The number of folds for *k*-fold cross-validation.
            If omitted,
            :attr:`~culebra.fitness_func.dataset_score.abc.ClassificationScorer._default_cv_num_folds`
            is used. Defaults to :data:`None`
        :type cv_num_folds: int
        :param cv_fixed_folds: If :data:`True`, the same folds are used across
            all evaluations, ensuring deterministic results. If :data:`False`,
            new folds are generated randomly for each evaluation, introducing
            variability in the results. If omitted,
            :attr:`~culebra.fitness_func.dataset_score.abc.ClassificationScorer._default_cv_fixed_folds`
            is used. Defaults to :data:`None`
        :type cv_fixed_folds: bool
        :param classifier: The classifier. If omitted,
            :attr:`~culebra.fitness_func.dataset_score.abc.ClassificationScorer._default_classifier`
            will be used. Defaults to :data:`None`
        :type classifier: ~sklearn.base.ClassifierMixin
        :param index: Index of this objective when it is used for
            multi-objective fitness functions, optional
        :type index: int
        :raises RuntimeError: If the number of objectives is not 1
        :raises TypeError: If *training_data* or *test_data* is an invalid
            dataset
        :raises TypeError: If *cv_num_folds* is not an integer value
        :raises ValueError: If *cv_num_folds* is not positive
        :raises TypeError: If *cv_fixed_folds* is not boolean
        :raises TypeError: If *classifier* is not a valid classifier
        :raises TypeError: If *index* is not an integer number
        :raises ValueError: If *index* is not positive
        """
        # Init the superclass
        super().__init__(
            training_data,
            test_data,
            cv_num_folds,
            cv_fixed_folds,
            index
        )
        self.classifier = classifier

    @property
    def _default_classifier(self) -> ClassifierMixin:
        """Default classifier.

        :return: A Gaussian Naive Bayes classifier
        :rtype: ~sklearn.base.ClassifierMixin
        """
        return GaussianNB()

    @property
    def classifier(self) -> ClassifierMixin:
        """Classifier applied within this fitness function.

        :rtype: ~sklearn.base.ClassifierMixin
        :setter: Set a new classifier
        :param value: The classifier. If set to :data:`None`,
            :attr:`~culebra.fitness_func.dataset_score.abc.ClassificationScorer._default_classifier`
            is used
        :type value: ~sklearn.base.ClassifierMixin
        :raises TypeError: If *value* is not a valid classifier
        """
        return self._classifier

    @classifier.setter
    def classifier(self, value: ClassifierMixin | None) -> None:
        """Set a classifier.

        :param value: The classifier. If set to :data:`None`,
            :attr:`~culebra.fitness_func.dataset_score.abc.ClassificationScorer._default_classifier`
            is used
        :type value: ~sklearn.base.ClassifierMixin
        :raises TypeError: If *value* is not a valid classifier
        """
        self._classifier = (
            self._default_classifier if value is None else check_instance(
                value, "classifier", cls=ClassifierMixin
            )
        )

    def _evaluate_train_test(
        self,
        sol: Solution,
        training_data: Dataset,
        test_data: Dataset
    ) -> tuple[float, ...]:
        """Evaluate a solution.

        Neither the solution nor its fitness should be modified.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :param training_data: The training dataset
        :type training_data: ~culebra.tools.Dataset
        :param test_data: The test dataset
        :type test_data: ~culebra.tools.Dataset
        :return: The fitness values for *sol*
        :rtype: tuple[float, ...]
        """
        outputs_pred = self.classifier.fit(
            training_data.inputs,
            training_data.outputs
        ).predict(test_data.inputs)

        return (type(self)._score(test_data.outputs, outputs_pred),)

    def _evaluate_kfcv(
        self,
        sol: Solution,
        training_data: Dataset,
    ) -> tuple[float, ...]:
        """Evaluate a solution.

        Neither the solution nor its fitness should be modified.

        A *k*-fold cross-validation is applied using the *training_data* with
        :attr:`~culebra.fitness_func.dataset_score.abc.ClassificationScorer.cv_num_folds`
        folds.

        :param sol: Solution to be evaluated.
        :type sol: ~culebra.abc.Solution
        :param training_data: The training dataset
        :type training_data: ~culebra.tools.Dataset
        :return: The fitness values for *sol*
        :rtype: tuple[float, ...]
        """
        scores = cross_val_score(
            self.classifier,
            training_data.inputs,
            training_data.outputs,
            cv=self.cv_splitter,
            scoring=make_scorer(self.__class__._score)
        )

        return(scores.mean(),)


# Exported symbols for this module
__all__ = [
    'DatasetScorer',
    'ClassificationScorer'
]
